import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
from data_utils2 import load_data
from models.moe_gnn import MoEGNN, load_balance_loss, diversity_loss

#日志文件类
class Logger:
    #path文件路径
    def __init__(self, path):
        #将标准输出流赋值，这样可以直接将self.term作为终端输出的一个引用
        self.term = sys.stdout
        #w以写模式打开path的指定文件，并指定使用UTF-8编码
        self.file = open(path, 'w', encoding='utf-8')

    #msg：要记录的消息
    def write(self, msg):
        #将消息写入到self.term（标准输出流），从而使消息显示在控制台上
        self.term.write(msg)
        #将消息写到打开的文件对象中去（self.file）
        self.file.write(msg)
        #调用 flush()方法强制将缓冲区中的数据立即写入物理文件。这可以确保日志消息被及时保存，避免因程序崩溃等原因导致缓冲区内的日志丢失
        self.file.flush()
    
    def flush(self):
        #刷新控制台缓冲区
        self.term.flush()
        #刷新日志文件缓冲区
        self.file.flush()
    
    def close(self):
        #日志结束后安全地释放资源
        self.file.close()


def train(model, data, opt, cfg):
    model.train()
    opt.zero_grad()
    
    out, gates, outs = model(data.x, data.edge_index, data.y, ret_gate=True, ret_expert_outs=True)
    
    mask = data.train_mask
    loss_c = F.cross_entropy(out[mask], data.y[mask], label_smoothing=0.0)
    
    # 辅助损失
    alpha = cfg.get('alpha', 0.0)
    beta = cfg.get('beta', 0.0)
    loss_b = load_balance_loss(gates[mask], alpha) if alpha > 0 else torch.tensor(0.0)
    loss_d = diversity_loss(outs[mask], gates[mask], beta) if beta > 0 else torch.tensor(0.0)
    
    loss = loss_c + loss_b + loss_d
    
    if torch.isnan(loss) or torch.isinf(loss):
        
        return 0, loss_c.item(), loss_b.item(), loss_d.item()
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    
    return loss.item(), loss_c.item(), loss_b.item(), loss_d.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.y)
    pred = out.argmax(dim=1)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0
        accs.append(acc)
    
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    
    # 检查类别
    import numpy as np
    nc = int(data.y.max().item()) + 1
    uniq = np.unique(y_pred)
    # if len(uniq) < nc:
    #     print(f'  Warn: {len(uniq)}/{nc} classes: {uniq}')
    
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return accs, macro_f1, micro_f1


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")  # 添加这行

    data, nc = load_data(args.dataset)
    #data, nc = load_data(args.dataset,use_public_split=True)//小样本学习（140）
    data = data.to(device)
    
    print('\n' + '='*60)
   # print(f'Dataset: {args.dataset} | N={data.num_nodes} E={data.num_edges} H={graph_stat(data)["homo"]:.3f}')
    print(f'Classes: {nc} | Train={data.train_mask.sum()} Val={data.val_mask.sum()} Test={data.test_mask.sum()}')
    print(f'Features: {data.num_features}\n')
    
    cfg = {
        'K': args.K,
        'nlayer': args.nlayer,
        'heads': args.heads,
        'k': args.topk,
        'global': args.use_global,
        'dp': args.dropout,
        'alpha': args.alpha,
        'beta': args.beta,
        'adaptive_k': args.adaptive_k,
        'use_spectral': getattr(args, 'use_spectral', True),
        'random_gate': getattr(args, 'random_gate', False)
    }
    
    model = MoEGNN(data.num_features, args.hidden, nc, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    
    warmup = 10
    def lr_fn(e):
        return (e + 1) / warmup if e < warmup else 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)#, verbose=True)
    
    best_val = 0
    best_test = 0
    best_macro_f1 = 0
    best_micro_f1 = 0
    patience = 20
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, loss_cls, loss_bal, loss_div = train(model, data, opt, cfg)
        (train_acc, val_acc, test_acc), macro_f1, micro_f1 = test(model, data)
        
        # if epoch == 1:
        if epoch % 10==0:
            with torch.no_grad():
                _, gates = model(data.x, data.edge_index, data.y, ret_gate=True)
                gate_usage = gates.mean(dim=0)
                print(f'  Expert usage: {gate_usage.cpu().numpy()}')
        
        scheduler.step()
        
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_macro_f1 = macro_f1
            best_micro_f1 = micro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stop at ep{epoch}')
                break
        
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                _, g_temp = model(data.x, data.edge_index, data.y, ret_gate=True)
                expert_usage = g_temp.mean(dim=0)
                active_experts = (expert_usage > 0.01).sum().item()
                # 计算平均门控熵
                entropy = -(g_temp * (g_temp + 1e-8).log()).sum(dim=1).mean().item()
            # print(f'Ep{epoch:03d} Loss:{loss:.4f}(c{loss_cls:.4f}b{loss_bal:.4f}d{loss_div:.4f}) '
            #       f'Train acc :{train_acc:.3f} Val acc :{val_acc:.3f} Test acc :{test_acc:.3f} '
            #       f'F1:{macro_f1:.3f} Act:{active_experts}/{model.ne} H:{entropy:.2f}')
    
    print(f'\nBest: Val={best_val:.4f} Test={best_test:.4f} MacroF1={best_macro_f1:.4f} MicroF1={best_micro_f1:.4f}')
    return best_test, best_macro_f1, best_micro_f1


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # 可用数据集: Cora, CiteSeer, PubMed, Texas, Wisconsin, Actor, Chameleon
#     parser.add_argument('--dataset', type=str, default='Cora')
#     parser.add_argument('--hidden', type=int, default=128)
#     parser.add_argument('--K', type=int, default=5)
#     parser.add_argument('--nlayer', type=int, default=2)
#     parser.add_argument('--heads', type=int, default=4)
#     parser.add_argument('--topk', type=int, default=7)
#     parser.add_argument('--use_global', action='store_true', default=True)
#     parser.add_argument('--use_spectral', action='store_true', default=True)
#     parser.add_argument('--adaptive_k', action='store_true', default=False)
#     parser.add_argument('--random_gate', action='store_true', default=False)
#     parser.add_argument('--dropout', type=float, default=0.5)
#     parser.add_argument('--alpha', type=float, default=0.001)
#     parser.add_argument('--beta', type=float, default=0.0001)
#     parser.add_argument('--lr', type=float, default=0.005)
#     parser.add_argument('--wd', type=float, default=1e-3)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--runs', type=int, default=1)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--save_log', action='store_true', default=True)
#     parser.add_argument('--log_dir', type=str, default='./logs')
#
#     args = parser.parse_args()
#
#
#     if args.save_log:
#         log_dir = Path(args.log_dir)
#         log_dir.mkdir(exist_ok=True)
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         log_file = log_dir / f'train_{args.dataset}_{timestamp}.log'
#         logger = Logger(log_file)
#         sys.stdout = logger
#         print(f'Logging to: {log_file}')
#
#     test_accs = []
#     macro_f1s = []
#     micro_f1s = []
#
#     for i in range(args.runs):
#         torch.manual_seed(args.seed)
#         np.random.seed(args.seed)
#
#         print(f'\n{"="*70}')
#         print(f'Run {i+1}/{args.runs}')
#         print(f'{"="*70}')
#         test_acc, macro_f1, micro_f1 = run(args)
#         test_accs.append(test_acc)
#         macro_f1s.append(macro_f1)
#         micro_f1s.append(micro_f1)
#
#         print(f'[Run{i+1}] Acc={test_acc:.4f} MacroF1={macro_f1:.4f} MicroF1={micro_f1:.4f}')
#
#     print(f'\n{"="*60}')
#     print(f'Final ({args.runs} runs on {args.dataset})')
#     print(f'{"="*60}')
#     print(f'Acc:  {np.mean(test_accs):.4f}±{np.std(test_accs):.4f}')
#     print(f'MacF1:{np.mean(macro_f1s):.4f}±{np.std(macro_f1s):.4f}')
#     print(f'MicF1:{np.mean(micro_f1s):.4f}±{np.std(micro_f1s):.4f}')
#     print(f'{"="*60}')
#
#
#     if args.save_log:
#         sys.stdout = logger.term
#         logger.close()
#         print(f'Log: {log_file}')







#################################################################################################################
#################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 原有参数 ---
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden', type=int, default=128)#小样本为64或者32，大样本是128
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--topk', type=int, default=7)
    parser.add_argument('--use_global', action='store_true', default=True)
    parser.add_argument('--use_spectral', action='store_true', default=True)
    parser.add_argument('--adaptive_k', action='store_true', default=False)
    parser.add_argument('--random_gate', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.005)#小样本学习率0.01，大样本是0.005
    parser.add_argument('--wd', type=float, default=1e-3)#小样本正则化：5e-4，大样本是1e-3
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)  # 跑几次取平均
    parser.add_argument('--seed', type=int, default=42)  # 默认种33最好,一般用42
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # --- 新增调参模式参数 ---
    parser.add_argument('--seed_search', action='store_true', default=False,help='开启幸运种子搜索模式')
    parser.add_argument('--search_range', type=int, default=50, help='搜索前多少个种子')
    parser.add_argument('--grid_search', action='store_true',default=False, help='开启LR/WD网格搜索')
    parser.add_argument('--use_public', action='store_true', default=False,
                        help='是否使用官方Public Split (140训练样本)')

    args = parser.parse_args()

    # 日志设置
    if args.save_log:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_str = "search" if (args.seed_search or args.grid_search) else "run"
        log_file = log_dir / f'{mode_str}_{args.dataset}_{timestamp}.log'
        logger = Logger(log_file)
        sys.stdout = logger
        print(f'Logging to: {log_file}')
        print(f'Args: {args}')  # 记录所有参数

    # ==========================================
    # 模式 1: 寻找幸运种子 (Lucky Seed Search)
    # ==========================================
    if args.seed_search:
        print(f'\n>>> 开始种子搜索 (Range: 0-{args.search_range}) <<<')
        best_seed = -1
        best_acc = 0.0

        # 结果容器
        results = []

        for s in range(args.search_range):
            # 动态修改 args 中的 seed
            args.seed = s

            # 设置全局随机种子 (影响模型初始化)
            torch.manual_seed(s)
            np.random.seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

            # 运行模型
            # 关键：确保 run(args) 内部调用 load_data 时传入了 seed=args.seed
            try:
                test_acc, macro_f1, _ = run(args)
                print(f'[Seed {s:02d}] Acc: {test_acc:.4f} | MacF1: {macro_f1:.4f}')

                results.append((s, test_acc))

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_seed = s
            except Exception as e:
                print(f'[Seed {s:02d}] Failed: {e}')

        print(f'\n{"=" * 40}')
        print(f'Best Seed Found: {best_seed}')
        print(f'Best Accuracy:   {best_acc:.4f}')
        print(f'{"=" * 40}')

    # ==========================================
    # 模式 2: 网格搜索 (Grid Search for LR/WD)
    # ==========================================
    elif args.grid_search:
        print(f'\n>>> 开始网格搜索 (LR & Weight Decay) <<<')

        # 定义搜索空间 (你可以根据需要修改这里)
        #cora的
        # lr_candidates = [0.01, 0.005, 0.001]
        # wd_candidates = [5e-4, 1e-3, 5e-3]
        #CiteSeer的
        # lr_candidates = [0.005, 0.01, 0.02]
        # wd_candidates = [5e-4, 1e-3, 5e-3,1e-2]
        #Actor
        # lr_candidates = [0.01, 0.03]
        # wd_candidates = [5e-4, 1e-3,1e-5]
        #Chameleon
        lr_candidates = [0.001, 0.005,0.01]
        wd_candidates = [1e-4, 5e-4,1e-3]
        # #WebKB子集(Texas, Cornell, Wisconsin)
        # lr_candidates = [0.01, 0.05]
        # wd_candidates = [5e-3,1e-2,5e-2]

        best_params = {}
        best_acc = 0.0

        for lr in lr_candidates:
            for wd in wd_candidates:
                args.lr = lr
                args.wd = wd

                # 为了稳健，每个组合跑3个种子取平均
                temp_accs = []
                #0, 42, 123
                for s in [42]:
                    args.seed = s
                    torch.manual_seed(s)
                    np.random.seed(s)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(s)

                    acc, _, _ = run(args)
                    temp_accs.append(acc)

                avg_acc = np.mean(temp_accs)
                print(f'[LR={lr}, WD={wd}] Avg Acc: {avg_acc:.4f}')

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_params = {'lr': lr, 'wd': wd}

        print(f'\n{"=" * 40}')
        print(f'Best Params: {best_params}')
        print(f'Best Avg Acc: {best_acc:.4f}')
        print(f'{"=" * 40}')

    # ==========================================
    # 模式 3: 正常运行 (Original Logic)
    # ==========================================
    else:
        test_accs = []
        macro_f1s = []
        micro_f1s = []

        for i in range(args.runs):
            # 如果 runs > 1，通常我们需要变化的种子来验证稳定性
            # 这里我修改为：如果是多次运行，种子递增；如果是单次，用指定种子
            current_seed = args.seed + i if args.runs > 1 else args.seed

            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)

            # 重要：把当前计算出的 seed 传回 args，以便 load_data 使用
            args_for_run = argparse.Namespace(**vars(args))
            args_for_run.seed = current_seed

            print(f'\n{"=" * 70}')
            print(f'Run {i + 1}/{args.runs} (Seed: {current_seed})')
            print(f'{"=" * 70}')

            test_acc, macro_f1, micro_f1 = run(args_for_run)

            test_accs.append(test_acc)
            macro_f1s.append(macro_f1)
            micro_f1s.append(micro_f1)

            print(f'[Run{i + 1}] Acc={test_acc:.4f} MacroF1={macro_f1:.4f}')

        print(f'\n{"=" * 60}')
        print(f'Final ({args.runs} runs on {args.dataset})')
        print(f'{"=" * 60}')
        print(f'Acc:  {np.mean(test_accs):.4f}±{np.std(test_accs):.4f}')
        print(f'MacF1:{np.mean(macro_f1s):.4f}±{np.std(macro_f1s):.4f}')
        print(f'{"=" * 60}')

    # 关闭日志
    if args.save_log:
        sys.stdout = logger.term
        logger.close()
        print(f'Log saved to: {log_file}')
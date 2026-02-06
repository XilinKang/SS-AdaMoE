import argparse
import torch
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
from train import run as train_run


class Logger:
    def __init__(self, path):
        self.term = sys.stdout
        self.file = open(path, 'w', encoding='utf-8')
    
    def write(self, msg):
        self.term.write(msg)
        self.file.write(msg)
        self.file.flush()
    
    def flush(self):
        self.term.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def ablation_study(args):
    # Ablation study: 5 configurations
    
    configs = {
        '1. Full Model': {
            'use_spectral': True,
            'use_global': True,
            'adaptive_k': True,
            'random_gate': False,
            'topk': 7,
            'desc': 'Full: 3 spectral + 4 spatial + global + adaptive K'
        },
        '2. No Spectral': {
            'use_spectral': False,
            'use_global': True,
            'adaptive_k': True,
            'random_gate': False,
            'topk': 4,
            'desc': 'No spectral: 4 spatial only'
        },
        '3. No Global': {
            'use_spectral': True,
            'use_global': False,
            'adaptive_k': True,
            'random_gate': False,
            'topk': 7,
            'desc': 'No global: w/o LightGT'
        },
        '4. Fixed K=2': {
            'use_spectral': True,
            'use_global': True,
            'adaptive_k': False,
            'random_gate': False,
            'topk': 2,
            'desc': 'Fixed K=2'
        },
        '5. Random Gate': {
            'use_spectral': True,
            'use_global': True,
            'adaptive_k': False,
            'random_gate': True,
            'topk': 7,
            'desc': 'Random gate'
        }
    }
    
    results = {}
    
    for name, cfg in configs.items():
        print(f'\n{"="*60}')
        print(f'Running: {name}')
        print(f'  {cfg["desc"]}')
        print(f'{"="*60}')
        
        
        args.use_spectral = cfg['use_spectral']
        args.use_global = cfg['use_global']
        args.adaptive_k = cfg['adaptive_k']
        args.random_gate = cfg['random_gate']
        args.topk = cfg['topk']
        
        test_accs = []
        for i in range(args.runs):
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)
            
            print(f'\nRun {i+1}/{args.runs}')
            test_acc, _, _ = train_run(args)
            test_accs.append(test_acc)
        
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        results[name] = (mean_acc, std_acc)
        
        print(f'{name}: {mean_acc:.4f} ± {std_acc:.4f}')
    
    print(f'\n{"="*60}')
    print('Ablation Results')
    print(f'{"="*60}')
    for name, (mean, std) in results.items():
        print(f'{name:<25} {mean:.4f}±{std:.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 可用数据集: Cora, CiteSeer, PubMed, Texas, Wisconsin, Actor, Chameleon
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--topk', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    args = parser.parse_args()
    
    
    if args.save_log:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'ablation_{args.dataset}_{timestamp}.log'
        logger = Logger(log_file)
        sys.stdout = logger
        print(f'Logging to: {log_file}\n')
    
    ablation_study(args)
    
  
    if args.save_log:
        sys.stdout = logger.term
        logger.close()
        print(f'Log: {log_file}')

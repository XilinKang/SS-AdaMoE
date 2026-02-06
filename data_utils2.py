import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork


# --- 新增：专门用于读取你本地原始Cora文件的函数 ---
def load_cora_manual(root_dir):
    print("正在从本地原始文件加载 Cora...")

    # 1. 确定文件路径
    # 假设你的目录结构是 datasets/Cora/cora.content
    content_path = os.path.join(root_dir, 'Cora', 'cora.content')
    cites_path = os.path.join(root_dir, 'Cora', 'cora.cites')

    if not os.path.exists(content_path):
        raise FileNotFoundError(f"找不到文件: {content_path}，请确认文件名和路径是否正确。")

    # 2. 读取节点特征 (cora.content)
    # 格式: <paper_id> <word_attributes>+ <class_label>
    raw_content = pd.read_csv(content_path, sep='\t', header=None)

    # 提取 ID (第一列)
    idx_map = {j: i for i, j in enumerate(raw_content.iloc[:, 0].values)}

    # 提取特征 (中间列) -> 转为 Tensor
    features = torch.FloatTensor(raw_content.iloc[:, 1:-1].values)

    # 提取标签 (最后一列) -> 转为索引
    labels_raw = raw_content.iloc[:, -1].values
    class_map = {label: i for i, label in enumerate(np.unique(labels_raw))}
    labels = torch.LongTensor([class_map[l] for l in labels_raw])

    # 3. 读取引用关系 (cora.cites)
    # 格式: <cited> <citing>
    raw_cites = pd.read_csv(cites_path, sep='\t', header=None)

    # 将原来的 Paper ID 映射为 0~2707 的索引
    # 注意：数据集中可能存在 source 或 target ID 不在 content 中的情况，需要过滤
    edges_list = []
    for _, row in raw_cites.iterrows():
        cited = row[0]
        citing = row[1]
        if cited in idx_map and citing in idx_map:
            # 注意方向：通常 Cora 视作无向图，或者引用方向。这里构建 source->target
            edges_list.append([idx_map[citing], idx_map[cited]])

    edge_index = torch.LongTensor(edges_list).t().contiguous()

    # 4. 构建 Data 对象
    data = Data(x=features, edge_index=edge_index, y=labels)
    num_classes = len(class_map)

    print(f"本地加载完成: 节点={data.num_nodes}, 边={data.num_edges}, 类别={num_classes}")
    return data, num_classes


# --- 修改后的主加载函数 ---
def load_data(name, root='./datasets', split_ratio=(0.6, 0.2, 0.2), seed=None):
    # 确保路径存在
    os.makedirs(root, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # === 分支 1: 如果是 Cora，直接读本地文件 ===
    if name == 'Cora':
        # 调用上面写的自定义函数
        data, num_classes = load_cora_manual(root)

    # === 分支 2: 其他数据集保持原样 (如果不需要它们可以不管) ===
    elif name in ['CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
    elif name in ['Texas', 'Wisconsin']:
        dataset = WebKB(root=root, name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
    elif name == 'Actor':
        dataset = Actor(root=root)
        data = dataset[0]
        num_classes = dataset.num_classes
    elif name == 'Chameleon':
        dataset = WikipediaNetwork(root=root, name='chameleon')
        data = dataset[0]
        num_classes = dataset.num_classes
    else:
        raise ValueError(f'Unknown dataset: {name}')

    # --- 统一的预处理步骤 ---
    # 1. 添加自环
    if data.edge_index is not None:
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # 2. 特征归一化
    data.x = F.normalize(data.x, p=2, dim=1)

    # 3. 生成 60/20/20 随机划分
    n = data.num_nodes
    idx = torch.randperm(n)
    n_train = int(split_ratio[0] * n)
    n_val = int(split_ratio[1] * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)

    data.train_mask[idx[:n_train]] = True
    data.val_mask[idx[n_train:n_train + n_val]] = True
    data.test_mask[idx[n_train + n_val:]] = True

    return data, num_classes
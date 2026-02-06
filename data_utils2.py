import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork



def load_cora_manual(root_dir):

    content_path = os.path.join(root_dir, 'Cora', 'cora.content')
    cites_path = os.path.join(root_dir, 'Cora', 'cora.cites')

    if not os.path.exists(content_path):
        raise FileNotFoundError(f"notfound: {content_path}。")


    raw_content = pd.read_csv(content_path, sep='\t', header=None)

    idx_map = {j: i for i, j in enumerate(raw_content.iloc[:, 0].values)}

    features = torch.FloatTensor(raw_content.iloc[:, 1:-1].values)

    labels_raw = raw_content.iloc[:, -1].values
    class_map = {label: i for i, label in enumerate(np.unique(labels_raw))}
    labels = torch.LongTensor([class_map[l] for l in labels_raw])



    raw_cites = pd.read_csv(cites_path, sep='\t', header=None)

    edges_list = []
    for _, row in raw_cites.iterrows():
        cited = row[0]
        citing = row[1]
        if cited in idx_map and citing in idx_map:
            edges_list.append([idx_map[citing], idx_map[cited]])

    edge_index = torch.LongTensor(edges_list).t().contiguous()

    data = Data(x=features, edge_index=edge_index, y=labels)
    num_classes = len(class_map)

    print(f"本地加载完成: 节点={data.num_nodes}, 边={data.num_edges}, 类别={num_classes}")
    return data, num_classes



def load_data(name, root='./datasets', split_ratio=(0.6, 0.2, 0.2), seed=None):

    os.makedirs(root, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    if name == 'Cora':

        data, num_classes = load_cora_manual(root)


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

    if data.edge_index is not None:
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)


    data.x = F.normalize(data.x, p=2, dim=1)

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

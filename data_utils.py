import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, HGBDataset
from torch_geometric.utils import add_self_loops
import numpy as np


def load_data(name, root='./datasets', split_ratio=(0.6, 0.2, 0.2)):
    # 加载数据：60/20/20划分
    import os
    os.makedirs(root, exist_ok=True)

    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name, force_reload=False)
    elif name in ['Texas', 'Wisconsin']:
        dataset = WebKB(root=root, name=name, force_reload=False)
    elif name == 'Actor':
        dataset = Actor(root=root, force_reload=False)
    elif name == 'Chameleon':
        dataset = WikipediaNetwork(root=root, name='chameleon', force_reload=False)
    elif name in ['CoraML', 'acm']:
        dataset = HGBDataset(root=root, name=name, force_reload=False)
    else:
        raise ValueError(f'Unknown dataset: {name}')

    data = dataset[0]
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data.x = F.normalize(data.x, p=2, dim=1)  # L2归一化

    # 60/20/20划分
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

    return data, dataset.num_classes


def get_homo(data):
    ei = data.edge_index
    y = data.y
    same = sum(1 for i in range(ei.size(1)) if y[ei[0, i]] == y[ei[1, i]])
    return same / ei.size(1)


def graph_stat(data):
    n = data.num_nodes
    e = data.edge_index.size(1)
    return {'n': n, 'e': e, 'deg': e / n, 'homo': get_homo(data)}


if __name__ == '__main__':
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            data, nc = load_data(name)
            st = graph_stat(data)
            print(f'{name}: n={st["n"]}, e={st["e"]}, homo={st["homo"]:.3f}')
        except:
            print(f'{name}: failed')

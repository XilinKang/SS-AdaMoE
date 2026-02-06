import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCNExpert(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayer=2, dp=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dp = dp
        
        self.convs.append(GCNConv(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))
        
        for _ in range(nlayer-2):
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))
        
        self.convs.append(GCNConv(hid_dim, out_dim))
    
    def forward(self, x, ei):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, ei)
            x = F.relu(bn(x))
            x = F.dropout(x, p=self.dp, training=self.training)
        return self.convs[-1](x, ei)


class SAGEExpert(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayer=2, dp=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dp = dp
        
        self.convs.append(SAGEConv(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))
        
        for _ in range(nlayer-2):
            self.convs.append(SAGEConv(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))
        
        self.convs.append(SAGEConv(hid_dim, out_dim))
    
    def forward(self, x, ei):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, ei)
            x = F.relu(bn(x))
            x = F.dropout(x, p=self.dp, training=self.training)
        return self.convs[-1](x, ei)


class GATExpert(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayer=2, heads=4, dp=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dp = dp
        
        self.convs.append(GATConv(in_dim, hid_dim, heads=heads, dropout=0.6))
        self.bns.append(nn.BatchNorm1d(hid_dim * heads))
        
        for _ in range(nlayer-2):
            self.convs.append(GATConv(hid_dim*heads, hid_dim, heads=heads, dropout=0.6))
            self.bns.append(nn.BatchNorm1d(hid_dim * heads))
        
        self.convs.append(GATConv(hid_dim*heads, out_dim, heads=1, concat=False, dropout=0.6))
    
    def forward(self, x, ei):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, ei)
            x = F.elu(bn(x))
            x = F.dropout(x, p=self.dp, training=self.training)
        return self.convs[-1](x, ei)


class JKExpert(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayer=3, dp=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dp = dp
        
        self.convs.append(GCNConv(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))
        
        for _ in range(nlayer-1):
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))
        
        self.lin = nn.Linear(nlayer * hid_dim, out_dim)
    
    def forward(self, x, ei):
        hs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = F.relu(bn(x))
            x = F.dropout(x, p=self.dp, training=self.training)
            hs.append(x)
        
        return self.lin(torch.cat(hs, dim=1))

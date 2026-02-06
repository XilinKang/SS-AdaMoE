import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian


class JacobiConv(nn.Module):
    def __init__(self, K, a=0.5, b=0.5, use_pcd=True, gamma_prime=0.1):
        super().__init__()
        self.K = K
        self.a = a
        self.b = b
        self.use_pcd = use_pcd
        self.gamma_prime = gamma_prime

        self.alpha = nn.Parameter(torch.Tensor(K + 1))
        if use_pcd:
            self.gamma = nn.Parameter(torch.Tensor(K))
            self.beta = nn.Parameter(torch.Tensor(K + 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.alpha, 0.9, 1.1)
        if self.use_pcd:
            nn.init.constant_(self.gamma, self.gamma_prime)
            nn.init.uniform_(self.beta, 0.9, 1.1)

    def forward(self, x, edge_index, lmax=2.0):
        n = x.size(0)
        device = x.device  # 关键：获取输入设备

        # 计算拉普拉斯矩阵
        idx, val = get_laplacian(edge_index, normalization='sym', num_nodes=n)
        idx, val = idx.to(device), val.to(device)  # 移动到正确设备

        L = torch.sparse_coo_tensor(idx, val, (n, n))

        # 创建单位矩阵 - 确保在正确设备
        indices = torch.stack([torch.arange(n, device=device),
                               torch.arange(n, device=device)])
        values = torch.ones(n, device=device)
        I = torch.sparse_coo_tensor(indices, values, (n, n))

        L_scaled = (2.0 / lmax) * L - I

        # Jacobi多项式递归计算
        P_list = [x]  # P0 = x
        if self.K >= 1:
            part1 = ((self.a - self.b) / 2) * x
            part2 = ((self.a + self.b + 2) / 2) * torch.sparse.mm(L_scaled, x)
            P1 = part1 + part2
            P_list.append(P1)

        for k in range(2, self.K + 1):
            theta_k = (2 * k + self.a + self.b) * (2 * k + self.a + self.b - 1) / (2 * k * (k + self.a + self.b))
            theta_k_prime = (2 * k + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2) / (
                        2 * k * (k + self.a + self.b) * (2 * k + self.a + self.b - 2))
            theta_k_double_prime = (k + self.a - 1) * (k + self.b - 1) * (2 * k + self.a + self.b) / (
                        k * (k + self.a + self.b) * (2 * k + self.a + self.b - 2))

            P_prev1 = P_list[k - 1]
            P_prev2 = P_list[k - 2]
            term1 = theta_k * torch.sparse.mm(L_scaled, P_prev1)
            term2 = theta_k_prime * P_prev1
            term3 = theta_k_double_prime * P_prev2
            P_k = term1 + term2 - term3
            P_list.append(P_k)

        # 组合输出
        out = self.alpha[0] * P_list[0]
        for k in range(1, self.K + 1):
            if self.use_pcd:
                gamma_prod = torch.prod(self.gamma[:k])
                alpha_k = self.beta[k] * gamma_prod
            else:
                alpha_k = self.alpha[k]
            out += alpha_k * P_list[k]

        return out


class JacobiExpert(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, K=10, a=0.5, b=0.5, dropout=0.5, nlayer=2, use_pcd=True):
        super().__init__()
        self.dropout = dropout

        # 保持您原有的MLP结构
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))

        for _ in range(nlayer - 2):
            self.lins.append(nn.Linear(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))

        self.lins.append(nn.Linear(hid_dim, out_dim))

        # 添加Jacobi滤波器
        self.jacobi_conv = JacobiConv(K, a, b, use_pcd)

    def forward(self, x, edge_index):
        # 先通过MLP（保持您原有的前向传播逻辑）
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = lin(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        # 应用Jacobi频谱滤波器
        x = self.jacobi_conv(x, edge_index)
        return x
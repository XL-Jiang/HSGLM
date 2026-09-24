import torch
import torch.nn as nn
from .mamba import MambaConfig, Mamba
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class T_convolution(nn.Module):
    def __init__(self, kernel_size=3, out_channels=None, d=30):
        """
        Temporal convolution applied along the time axis (dim=-1) of each window.
        All windows and all ROIs share the SAME conv kernel (parameter sharing).
        """
        super(T_convolution, self).__init__()
        self.kernel_size = kernel_size
        self.d = d
        # Conv1d: (N, C_in, L) -> (N, C_out, L)
        # 输入通道为 1（单变量 BOLD 序列）
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=0,   # 左填充，因果卷积
            bias=True
        )
        self.proj = nn.Linear(d, d)
        self.elu = nn.ELU()
        self.out_channels = out_channels
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

    def forward(self, x):
        # x: (B, W, N, C)
        B, W, N, C = x.shape
        # 合并 (B, W, N) 作为 batch，C 作为序列长度，通道=1
        x_flat = x.reshape(B * W * N, 1, C)          # (B*W*N, 1, C)
        x_flat = F.pad(x_flat, (self.kernel_size - 1, 0))  # 因果填充
        x_conv = self.conv(x_flat)                    # (B*W*N, C_out, C)
        x_conv = x_conv.squeeze(1)  # (B*W*N, C)
        x_conv = self.elu(x_conv)
        # (B, W, N, C_out)
        x_feat = self.proj(x_conv)
        x_out = x_feat.reshape(B, W, N, C)
        return x_out

class SignedGCN(MessagePassing):
    """
    支持有符号边权重的 GCN 层，所有时间窗之间的权重共享
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        # 统一的特征变换权重（在所有时间窗和批次间共享）
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_self = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight):
        """
        x:          (B * W * N, in_dim)
        edge_index: (2, E)
        edge_weight: (E,) 可正可负的有符号权重
        """
        x_transformed =F.relu(self.lin(x))
        out = self.propagate(edge_index, x=x_transformed, edge_weight=edge_weight)
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, edge_index[0], torch.ones_like(edge_weight))
        out = out / deg.clamp(min=1).unsqueeze(-1)
        out = self.lin_self(out)
        return F.relu(out)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j

class DynamicGraphConstruction(nn.Module):
    def __init__(self, feat_dim, sparsity_ratio=30, eps=1e-6):  # 默认值改为 30
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.eps = eps
        self.scale = feat_dim ** 0.5
        # 可学习的 Q/K 投影：解耦秩-1
        self.W_Q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_K = nn.Linear(feat_dim, feat_dim, bias=False)

    def forward(self, x):
        """
        x: (B, W, N, P)
        return: edge_index, edge_weight  (有符号权重)
        """
        B, W, N, P = x.shape
        x_flat = x.reshape(B * W, N, P)              # (BW, N, P)
        q = self.W_Q(x_flat)                          # (BW, N, P)
        k = self.W_K(x_flat)                          # (BW, N, P)
        S = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (BW, N, N)
        S = 0.5 * (S + S.transpose(1, 2))
        eye = torch.eye(N, device=S.device).unsqueeze(0)
        S = S + eye
        k_top = max(1, int(N * self.sparsity_ratio / 100.0))
        _, idx = torch.topk(S.abs(), k_top, dim=-1)
        mask = torch.zeros_like(S).scatter_(-1, idx, 1.0)
        S = S * mask
        S = 0.5 * (S + S.transpose(1, 2))
        ei_list, ew_list = [], []
        for i in range(B * W):
            rows, cols = torch.where(S[i] != 0)
            offset = i * N
            ei_list.append(torch.stack([rows + offset, cols + offset], 0))
            ew_list.append(S[i][rows, cols])
        ei = torch.cat(ei_list, 1)
        ew = torch.cat(ew_list, 0)
        return ei, ew


class GCMRP(nn.Module):
    """
      1)  双通道池化
      2) 使用 top-K 加权求和，而非对所有 ROI 平均
      3) 注意力分数由全局上下文与节点特征共同决定（不是简单 MLP）
    """
    def __init__(self,feat_dim,top_alpha, dropout=0.3):
        super(GCMRP, self).__init__()
        self.top_alpha = top_alpha
        # 全局上下文 -> 查询向量
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        # 节点特征 -> 键向量
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, W, N, P)
        return:
            out: (B, W, P)   —— 固定维度，可供跨图谱拼接
            attn: (B, W, N)  —— 完整注意力权重，用于可视化
        """
        B, W, N, P = x.shape

        gap = x.mean(dim=2)                  # (B, W, P)
        gmp = x.max(dim=2)[0]                # (B, W, P)
        global_ctx = 0.5 * (gap + gmp)       # (B, W, P)


        q = self.q_proj(global_ctx)          # (B, W, P)
        k = self.k_proj(x)                   # (B, W, N, P)
        # 内积打分 (B, W, N)
        score = (q.unsqueeze(2) * k).sum(dim=-1) / (P ** 0.5)
        attn = torch.sigmoid(score)          # (B, W, N)
        attn = self.dropout(attn)

        # 3) Top_alpha选择 + 加权求和 -> (B, W, P)
        if self.top_alpha == -1:
            weights = attn.unsqueeze(-1)  # (B, W, N, 1)
            out = (x * weights).sum(dim=2)  # (B, W, P)
        else:
            top_val, top_idx = torch.topk(attn, self.top_alpha, dim=-1)  # (B, W, K)
            idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, -1, P)  # (B, W, K, P)
            selected = torch.gather(x, 2, idx_expand)  # (B, W, K, P)
            weights = top_val.unsqueeze(-1)  # (B, W, K, 1)
            out = (selected * weights).sum(dim=2)  # (B, W, P)

        return out, attn


class ModuleMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim,n_layers):
        super().__init__()
        self.config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.mamba = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            Mamba(self.config),
            # nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.mamba(x)
        return x

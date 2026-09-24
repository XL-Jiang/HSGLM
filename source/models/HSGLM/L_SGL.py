import torch
import torch.nn as  nn
from source.models.HSGLM.components import T_convolution, DynamicGraphConstruction
class L_SGL(nn.Module):
    def __init__(self,window_length,sparsity_beta=0.3,out_channels=None):
        super(L_SGL, self).__init__()
        self.TConv = T_convolution(out_channels=out_channels or window_length,d=window_length)
        self.DGConstruction = DynamicGraphConstruction(feat_dim=out_channels or window_length,sparsity_ratio=sparsity_beta)
        self.model_init()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    def forward(self, dtimeseries):
        """
         Args:
            dtimeseries: (B, W, N, C)  —— 原始 BOLD 时间序列
                         B = batch, W = 窗口数, N = ROI 数, C = 窗口长度
        Returns:
            t_conv:    (B, W, N, out_channels)  —— T-conv 后的节点特征
            edge_index: (2, E)  —— 有符号边的全局索引
            edge_weight: (E,)   —— 有符号边权重，可正可负

        """
        #T-conv
        t_conv = self.TConv(dtimeseries) #b w n t
        edge_index, edge_weight = self.DGConstruction(t_conv)
        return t_conv, edge_index, edge_weight
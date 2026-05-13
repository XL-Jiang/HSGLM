from Module import  *
from einops import rearrange, repeat

class L_SGL(nn.Module):
    """
    L_SGL (Local spatiotemporal graph learning) model.
    """
    def __init__(self,hidden_dim, num_time,window_size, kernel_size):
        super(L_SGL, self).__init__()
        self.initial_linear = nn.Linear(window_size, hidden_dim)
        self.TConv  = T_convolution(kernel_size = kernel_size, inc=num_time)
        self.DGConstruction = DynamicGraphConstruction()
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
        # assumes shape [minibatch x windows x  time  x node ] for dtimeseries
        """
        minibatch_size, num_windows,num_timepoints,num_nodes = dtimeseries.shape[:4]

        #T-conv
        t = rearrange(dtimeseries, 'b w t n -> b w n t')
        t_conv = self.TConv(t) #b w n t
        dy_ftrs,dy_adjs = self.DGConstruction(t_conv)# (b w) n t,(b w) n n
        dy_local = self.initial_linear(dy_ftrs)#(b w) n hidden
        dy_local = rearrange(dy_local, '(b w) n h -> b w n h', w=num_windows, b=minibatch_size, n=num_nodes)

        return dy_local,dy_adjs

import torch
import torch.nn as nn
from .L_SGL import L_SGL
from .G_SGL import G_SGL
from omegaconf import DictConfig
class HSGLM(nn.Module):
    def __init__(self, config: DictConfig):
        super(HSGLM, self).__init__()
        self.hidden_dim = config.model.hidden_dim
        self.window_length = config.dataset.timeseries_sz
        self.num_windows = config.dataset.windows_sz
        self.forward_dim = config.model.hidden_dim
        self.top_alpha = config.model.top_alpha
        self.sparsity_beta = config.model.sparsity_beta
        self.num_heads = config.model.num_heads
        self.phd_dim =config.dataset.non_imaging_feature_sz
        self.num_nodes1 = config.dataset.node_sz[0]
        self.num_nodes2 = config.dataset.node_sz[1]
        self.num_nodes3 = config.dataset.node_sz[2]

        self.L_SGL1= L_SGL(
                    window_length=self.window_length,
                    sparsity_beta=self.sparsity_beta)
        self.L_SGL2 = L_SGL(
                    window_length=self.window_length,
                    sparsity_beta=self.sparsity_beta)
        self.L_SGL3 = L_SGL(
                    window_length=self.window_length,
                    sparsity_beta=self.sparsity_beta)

        self.G_SGL= G_SGL(self.window_length,self.forward_dim,self.top_alpha,config.model.mamba_layers,self.num_heads)
        self.dim_reduction = nn.Sequential(
            nn.Linear(3*self.forward_dim, self.forward_dim),
        )
        self.mlp = nn.Sequential(nn.Linear(self.phd_dim, self.forward_dim), nn.LeakyReLU())
        # Classfier
        self.out = nn.Sequential(
            nn.Linear(2* self.forward_dim, self.forward_dim),
            nn.LeakyReLU(),
            nn.Linear(self.forward_dim, 2)
        )
        self.model_init()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, dtimesreies1,dtimesreies2,dtimesreies3,phd_ftrs):

        #L_SGL
        t1, ei1, ew1 = self.L_SGL1(dtimesreies1) # b w n1 n1, 2 E1
        t2, ei2, ew2 = self.L_SGL2(dtimesreies2)  # b w n2 n2, 2 E2
        t3, ei3, ew3 = self.L_SGL3(dtimesreies3)  # b w n3 n3, 2 E3
        #G_SGL
        dy_mambaout,attention = self.G_SGL(
        t1, t2, t3,
        ei1, ew1,
        ei2, ew2,
        ei3, ew3) #b w 3*forward_dim
        dy_mambaout = self.dim_reduction(dy_mambaout)#b w forward_dim
        dy_mambaout = dy_mambaout.max(dim=1)[0]  # B, forward_dim
        #process Non-imaging Data
        phd_out = self.mlp(phd_ftrs) #b phd_dim
        #cat fMRI & phd
        concatenated = torch.cat((dy_mambaout, phd_out), dim=1)
        # Classfier
        out = self.out(concatenated)#b 2
        return out,concatenated,attention


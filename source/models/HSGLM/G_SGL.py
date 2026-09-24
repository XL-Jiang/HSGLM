import  torch
import torch.nn as  nn
from source.models.HSGLM.components import GCMRP, ModuleMamba,InterpretableTransformerEncoder, SignedGCN


class G_SGL(nn.Module):

    def __init__(self, input_dim,forward_dim,top_alpha,mamba_layers,num_heads):
        super(G_SGL, self).__init__()
        self.forward_dim=forward_dim

        self.readout_module1 = GCMRP(feat_dim=forward_dim,top_alpha=top_alpha)
        self.readout_module2 = GCMRP(feat_dim=forward_dim, top_alpha=top_alpha)
        self.readout_module3 = GCMRP(feat_dim=forward_dim, top_alpha=top_alpha)

        # self.transformer = InterpretableTransformerEncoder(d_model=3*forward_dim, nhead=num_heads,dim_feedforward=2*3*forward_dim,batch_first=True)
        self.DGConvolution1 = SignedGCN(input_dim, forward_dim)
        self.DGConvolution2 = SignedGCN(input_dim, forward_dim)
        self.DGConvolution3 = SignedGCN(input_dim, forward_dim)
        self.mamba = ModuleMamba(3*forward_dim, 3*forward_dim, mamba_layers)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, t1, t2, t3,ei1, ew1, ei2, ew2, ei3, ew3):
        """
        # assumes shape [minibatch x window_size x node x hidden ] for dy_local
        # assumes shape [2   E] for dy_adjs

        """

        attention = {'node-attention1': [],'node-attention2': [],'node-attention3': []}
        B, W, N1, H = t1.shape[:4]
        _, _, N2, _ = t2.shape[:4]
        _, _, N3, _ = t3.shape[:4]

        # B, W, N to GCN
        x1 = t1.reshape(B * W * N1, H)
        x2 = t2.reshape(B * W * N2, H)
        x3 = t3.reshape(B * W * N3, H)

        ######## 1 *******
        # GCN1
        dy_GCNout1 = self.DGConvolution1(x1, ei1, ew1).reshape(B, W, N1,self.forward_dim)  # (b w n) hidden_dim

        #Readout
        dy_readout_topk1,node_attn1 = self.readout_module1(dy_GCNout1) #dy_readout,b w hidden_dim

        ######## 2 ********
        # GCN1
        dy_GCNout2 = self.DGConvolution2(x2, ei2, ew2).reshape(B, W, N2,self.forward_dim)  # (b w n) hidden_dim

        # Readout
        dy_readout_topk2, node_attn2 = self.readout_module2(dy_GCNout2)  # dy_readout,b w hidden_dim

        ######## 3 ********
        # GCN1
        dy_GCNout3 = self.DGConvolution3(x3, ei3, ew3).reshape(B, W, N3, self.forward_dim)  # (b w n) hidden_dim

        # Readout
        dy_readout_topk3, node_attn3 = self.readout_module3(dy_GCNout3)  #dy_readout,b w hidden_dim

        #Mamba
        dy_readout_topk = torch.cat((dy_readout_topk1,dy_readout_topk2,dy_readout_topk3),dim=-1)#b w 3hidden_dim

        dy_out = self.mamba(dy_readout_topk) #b w 3hidden_dim
        # dy_out = self.transformer(dy_readout_topk)  # b w 3hidden_dim

        attention = {
            'node-attention1': node_attn1.detach().cpu(),  # (B, W, N1)
            'node-attention2': node_attn2.detach().cpu(),  # (B, W, N2)
            'node-attention3': node_attn3.detach().cpu(),  # (B, W, N3)
        }

        return dy_out, attention
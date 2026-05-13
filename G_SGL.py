from Module import  *
from einops import rearrange, repeat

class G_SGL(nn.Module):
    """
    G_SGL (Global spatiotemporal graph learning) model.
    """
    def __init__(self, input_dim1, input_dim2,input_dim3,hidden_dim,topk,window_size,num_heads, num_time,cls_token='sum', readout='sero'):
        super(G_SGL, self).__init__()
        assert cls_token in ['sum', 'mean', 'max', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(1)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(1)
        elif cls_token == 'max':
            self.cls_token = lambda x: x.max(1)[0]
        elif cls_token == 'param':
            self.cls_token = lambda x: x[:, -1, :]
        else:
            raise ValueError("Invalid cls_token type")
        if readout == 'baro':
            self.readout_module1 = BARO(hidden_dim=hidden_dim, input_dim=input_dim1,topk=topk)
            self.readout_module2 = BARO(hidden_dim=hidden_dim, input_dim=input_dim2, topk=topk)
            self.readout_module3 = BARO(hidden_dim=hidden_dim, input_dim=input_dim3, topk=topk)
        elif readout == 'mean':
            self.readout_module  = ModuleMeanReadout
        else:
            raise ValueError("Invalid readout type")
        self.DGConvolution1 = GraphConvolution(nums_graph = num_time,in_features = window_size,out_features = hidden_dim)
        self.DGConvolution2 = GraphConvolution(nums_graph = num_time,in_features = window_size,out_features = hidden_dim)
        self.DGConvolution3 = GraphConvolution(nums_graph = num_time,in_features = window_size,out_features = hidden_dim)
        self.mamba = ModuleMamba(3*hidden_dim, 3*hidden_dim, 1)
        self.fc = nn.Sequential(nn.BatchNorm1d(3*hidden_dim), nn.Linear(3*hidden_dim, hidden_dim), nn.ReLU())
        self.model_init()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, dy_local1,dy_local2,dy_local3,dy_adjs1,dy_adjs2,dy_adjs3):
        """
        # assumes shape [minibatch x window_size x node x  hidden   ] for dy_local
        """
        attention = {'node-attention1': [],'node-attention2': [],'node-attention3': []}
        minibatch_size, num_windows,num_nodes1,_ = dy_local1.shape[:4]
        minibatch_size, num_windows, num_nodes2, _ = dy_local2.shape[:4]
        minibatch_size, num_windows, num_nodes3, _ = dy_local3.shape[:4]

        ######## 1 *******
        # GCN1
        dy_GCNout1 = self.DGConvolution1(dy_local1, dy_adjs1)  # (b w n) hidden_dim
        dy_GCNout1 = rearrange(dy_GCNout1, '(b w) n h -> b w n h', w=num_windows, b=minibatch_size, n=num_nodes1)

        #Readout
        dy_readout1, dy_readout_topk1,node_attn1 = self.readout_module1(dy_GCNout1) #dy_readout,b w h

        ######## 2 ********
        # GCN
        dy_GCNout2 = self.DGConvolution2(dy_local2, dy_adjs2)  # (b w n) hidden_dim
        dy_GCNout2 = rearrange(dy_GCNout2, '(b w) n h -> b w n h', w=num_windows, b=minibatch_size, n=num_nodes2)


        # Readout
        dy_readout2, dy_readout_topk2, node_attn2 = self.readout_module2(dy_GCNout2)  # dy_readout,b w kh

        ######## 3 ********
        # GCN
        dy_GCNout3 = self.DGConvolution3(dy_local3, dy_adjs3)  # (b w n) hidden_dim
        dy_GCNout3 = rearrange(dy_GCNout3, '(b w) n h -> b w n h', w=num_windows, b=minibatch_size, n=num_nodes3)


        # Readout
        dy_readout3, dy_readout_topk3, node_attn3 = self.readout_module3(dy_GCNout3)  # dy_readout,b w h

        #Mamba
        dy_readout_topk = torch.cat((dy_readout_topk1,dy_readout_topk2,dy_readout_topk3),dim=2)#b 3w h
        dy_mambaout = self.mamba(dy_readout_topk) #b w 3h
        dy_mambaout = self.cls_token(dy_mambaout)  # b 3h
        dy_mambaout = self.fc(dy_mambaout)  # b h

        attention['node-attention1'].append(node_attn1)
        attention['node-attention2'].append(node_attn2)
        attention['node-attention3'].append(node_attn3)
        attention['node-attention1'] = torch.stack([tensor[0] for tensor in attention['node-attention1']], dim=0).detach().cpu() #只取第一个张量
        attention['node-attention2'] = torch.stack([tensor[0] for tensor in attention['node-attention2']], dim=0).detach().cpu() #只取第一个张量
        attention['node-attention3'] = torch.stack([tensor[0] for tensor in attention['node-attention3']], dim=0).detach().cpu() #只取第一个张量


        return dy_mambaout, attention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, GCNConv, SGConv,  BatchNorm, GATConv # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool
from models2.explainer import Model
import numpy as np
class explain_train(torch.nn.Module):
    def __init__(self, feature, out_channel,pooltype,drop_rate,backbone, dropping_method,heads,K,alpha,first_layer_dimension,batch_size):
        super(explain_train, self).__init__()

        self.drop_rate = drop_rate
        self.backbone = backbone
        self.dropping_method = dropping_method
        self.heads = heads
        self.K = K
        self.alpha = alpha
        self.pooltype = pooltype
        self.first_layer_dimension = first_layer_dimension
        self.pool1, self.pool2 = self.poollayer(pooltype)
        self.model = Model()

    def forward(self, data, drop_rate):
        x, edge_index,batch= data.x, data.edge_index,data.batch
        if x.shape[1] == 500:
            x = x.narrow(1,20,480)
            # 假设每个图有5个节点
            num_nodes_per_graph = 35
            # 将 x 调整为 (batch_size, num_node_features, num_nodes)
            x = x.reshape(-1, 480, num_nodes_per_graph)
            new_shape = (x.shape[0], 1, x.shape[1], x.shape[2])
            x = x.reshape(new_shape)
            out= self.model(x)
            self.print_parameters()

        if x.shape[1] == 960:
            x = x.narrow(1, 160, 480)
            # 假设每个图有5个节点
            num_nodes_per_graph = 35
            # 将 x 调整为 (batch_size, num_node_features, num_nodes)
            x = x.reshape(-1, 480, num_nodes_per_graph)
            new_shape = (x.shape[0], 1, x.shape[1], x.shape[2])
            x = x.reshape(new_shape)
            out = self.model(x)

        x3 = F.log_softmax(out, dim = 1)
        x4 = torch.argmax(out, dim = 1)
        return out,x4,out
    def print_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')

    def poollayer(self, pooltype):

        self.pooltype = pooltype

        if self.pooltype == 'TopKPool':
            self.pool1 = TopKPooling(1024)
            self.pool2 = TopKPooling(1024)
        elif self.pooltype == 'EdgePool':
            self.pool1 = EdgePooling(1024)
            self.pool2 = EdgePooling(1024)
        elif self.pooltype == 'ASAPool':
            self.pool1 = ASAPooling(1024)
            self.pool2 = ASAPooling(1024)
        elif self.pooltype == 'SAGPool':
            self.pool1 = SAGPooling(1024)
            self.pool2 = SAGPooling(1024)
        else:
            print('Such graph pool method is not implemented!!')

        return self.pool1, self.pool2

    def poolresult(self,pool,pooltype,x,edge_index,batch):

        self.pool = pool

        if pooltype == 'TopKPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'EdgePool':
            x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'ASAPool':
            x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'SAGPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        else:
            print('Such graph pool method is not implemented!!')

        return x, edge_index, batch



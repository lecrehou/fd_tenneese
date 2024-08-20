from models2.inception import YourModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, GCNConv, SGConv,  BatchNorm, GATConv # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool
import torch.nn as nn

class inception_train(torch.nn.Module):
    def __init__(self, feature, out_channel,pooltype,drop_rate,backbone, dropping_method,heads,K,alpha,first_layer_dimension):
        super(inception_train, self).__init__()

        self.drop_rate = drop_rate
        self.backbone = backbone
        self.dropping_method = dropping_method
        self.heads = heads
        self.K = K
        self.alpha = alpha
        self.pooltype = pooltype
        self.first_layer_dimension = first_layer_dimension
        self.pool1, self.pool2 = self.poollayer(pooltype)
        # inception_time后添加的参数
        self.gap_layer = nn.AdaptiveAvgPool1d(1)
        self.model = YourModel((64,52,480), 21, 6, True, 32)
        self.to("cuda")




    def forward(self, data, drop_rate, pooltype):
        x, edge_index,batch= data.x, data.edge_index,data.batch
        if x.shape[1] == 500:
            x = x.narrow(1,20,480)
            num_nodes_per_graph = 52
            x = x.reshape(-1, num_nodes_per_graph,480)
#             x.to("cuda")
            out = self.model(x)
            out = self.gap_layer(out).squeeze()
            # 定义线性层
            out_layer = nn.Linear(out.size(1),21)
            out_flat = out_layer(out)

        if x.shape[1] == 960:
            x = x.narrow(1, 160, 480)
            num_nodes_per_graph = 52
            x = x.reshape(-1, num_nodes_per_graph, 480)

#
            out = self.model(x)
            out = self.gap_layer(out).squeeze()
            # 定义线性层
            out_layer = nn.Linear(out.size(1), 21)
            out_flat = out_layer(out)

        x3 = F.log_softmax(out_flat, dim = 1)
        x4 = torch.argmax(out_flat, dim = 1)
        return out_flat,x4,out_flat

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
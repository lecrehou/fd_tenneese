from models2.inception_time_new import Inception, InceptionBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, GCNConv, SGConv,  BatchNorm, GATConv # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool
import torch.nn as nn



class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)
class inception_time_new_train(torch.nn.Module):
    def __init__(self, feature, out_channel,pooltype,drop_rate,backbone, dropping_method,heads,K,alpha,first_layer_dimension,batch_size):
        super(inception_time_new_train, self).__init__()

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

        self.model = InceptionTime = nn.Sequential(
                    Reshape(out_shape=(52,480)),
                    InceptionBlock(
                        in_channels=52,
                        n_filters=32,
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    InceptionBlock(
                        in_channels=32*4,
                        n_filters=32,
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    Flatten(out_features=32*4*1),
                    nn.Linear(in_features=4*32*1, out_features=21)
        )




    def forward(self, data, drop_rate):
        x, edge_index,batch= data.x, data.edge_index,data.batch
        if x.shape[1] == 500:
            x = x.narrow(1,20,480)
            num_nodes_per_graph = 52
            x = x.reshape(-1, num_nodes_per_graph,480)
            out = self.model(x)




        if x.shape[1] == 960:
            x = x.narrow(1, 160, 480)
            num_nodes_per_graph = 52
            x = x.reshape(-1, num_nodes_per_graph, 480)

            out = self.model(x)


        x3 = F.log_softmax(out, dim = 1)
        x4 = torch.argmax(x3, dim = 1)
        return out,x4,out

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
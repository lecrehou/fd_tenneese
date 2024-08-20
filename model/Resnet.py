import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, GCNConv, SGConv, BatchNorm, GATConv  # noqa
from torch_geometric.nn import TopKPooling, EdgePooling, ASAPooling, SAGPooling, global_mean_pool
from models2.layers import GNNLayer
import numpy as np
import pdb

class Resnet(torch.nn.Module):
    def __init__(self, feature, out_channel, pooltype, drop_rate, backbone, dropping_method, heads, K, alpha,
                 first_layer_dimension, batch_size):
        super(Resnet, self).__init__()

        self.drop_rate = drop_rate
        self.backbone = backbone
        self.dropping_method = dropping_method
        self.heads = heads
        self.K = K
        self.alpha = alpha
        self.pooltype = pooltype
        self.first_layer_dimension = first_layer_dimension
        self.pool1, self.pool2 = self.poollayer(pooltype)
        self.block_1 = ResNetBlock(1, 64)
        self.block_2 = ResNetBlock(64, 64)
        self.block_3 = ResNetBlock(64, 64)


        self.dropout = nn.Dropout(0.2)
        self.gc = GATConv(64, 64)
        self.gc11 = GATConv(64, 21)

        self.gc1 = GNNLayer(64, first_layer_dimension, dropping_method, backbone, heads, K, alpha)
        self.gc2 = GNNLayer(first_layer_dimension * heads, first_layer_dimension * heads, dropping_method, backbone,
                            heads,K, alpha)
        self.gc3 = GNNLayer(first_layer_dimension * heads, 21, dropping_method, backbone, heads,K, alpha)

        self.bn2 = BatchNorm(64)
        self.batch_size = batch_size

    def forward(self, data, drop_rate):
        x, edge_index,batch = data.x, data.edge_index,data.batch
        if 'batch' not in data:
            batch = torch.tensor([0])
            batch = batch.to(x.device)  # 设置为 1
        else:
            batch = data.batch
            batch = batch.to(x.device)
        if x.shape[1] == 500:
            x = x.narrow(1, 20, 480)
            x = torch.tensor(x)
            x = torch.unsqueeze(x, axis=1)

            x = self.block_1(x)
            x = self.block_2(x)
            x = F.avg_pool1d(x, x.shape[-1]).squeeze()
            x = self.gc1(x, edge_index,drop_rate)

            x = x.view(self.batch_size,52,64)
            x = x.transpose(1, 2)
            x = self.bn2(x)
            x = x.transpose(2,1)
            x = x.reshape(self.batch_size*52, 64)

            x = F.relu(x)
            x = self.gc2(x, edge_index,drop_rate)
            x2 = global_mean_pool(x, batch)          
        if x.shape[1] == 960:
            x = x.narrow(1, 160, 480)
            x = torch.tensor(x)
            x = torch.unsqueeze(x, axis=1)

            x = self.block_1(x)
            x = self.block_2(x)

            x = F.avg_pool1d(x, x.shape[-1]).squeeze()
            x = self.gc1(x, edge_index, drop_rate)

            x = x.view(self.batch_size, 52, 64)
            x = x.transpose(1, 2)
            x = self.bn2(x)
            x = x.transpose(2, 1)
            x = x.reshape(self.batch_size * 52, 64)
            
            x = F.relu(x)
         
            x = self.gc2(x, edge_index, drop_rate)
            x2 = global_mean_pool(x, batch)




        x3 = F.log_softmax(x2, dim=1)
        x4 = torch.argmax(x2, dim=1)
        return x2, x4
#     def print_parameters(self):
#         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f'Total number of parameters: {total_params}')

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

    def poolresult(self, pool, pooltype, x, edge_index, batch):

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


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.expand = True if in_channels < out_channels else False

        self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, _, L = x.shape
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)

        return out
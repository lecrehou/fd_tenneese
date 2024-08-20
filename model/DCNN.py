import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, GCNConv, SGConv,  BatchNorm, GATConv # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool
from models2.layers import GNNLayer
import numpy as np
class DCNN(torch.nn.Module):
    def __init__(self, feature, out_channel,pooltype,drop_rate,backbone, dropping_method,heads,K,alpha,first_layer_dimension,batch_size):
        super(DCNN, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(30464, 300),
            # 原先是182784
            nn.Linear(182784,300),
            nn.Dropout(0.5))
        self.fc1 = nn.Sequential(nn.Linear(300, 21))

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2*2), stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2*1), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, drop_rate):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.shape[1] == 500:
            x = x.narrow(1, 20, 480)
            # 假设每个图有5个节点
            num_nodes_per_graph = 52
            x = x.reshape(-1, 480, num_nodes_per_graph)
            new_shape = (x.shape[0], 1, x.shape[1], x.shape[2])
            x = x.reshape(new_shape)
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x2 = self.pool1(x2)
            x3 = self.conv3(x2)
            x3 = self.pool2(x3)
            
            flatten = x3.reshape((64,-1))
            
            x4 = self.fc(flatten)
            out = self.fc1(x4)
            self.print_parameters()

        if x.shape[1] == 960:
            x = x.narrow(1, 160, 480)
            # 假设每个图有5个节点
            num_nodes_per_graph = 52
            # 将 x 调整为 (batch_size, num_node_features, num_nodes)
            x = x.reshape(-1, 480, num_nodes_per_graph)
            new_shape = (x.shape[0], 1, x.shape[1], x.shape[2])
            x = x.reshape(new_shape)
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x2 = self.pool1(x2)
            x3 = self.conv3(x2)
            x3 = self.pool2(x3)
            flatten = x3.reshape((64, -1))
           
            x4 = self.fc(flatten)
            out = self.fc1(x4)

        x3 = F.log_softmax(out, dim=1)
        x5 = torch.argmax(out, dim=1)
        return out, x5,out
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


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        self.expand = True if in_channels < out_channels else False
        self.conv_x = nn.Conv1d(in_channels, out_channels, kernel_size= 3, padding=1)
        self.conv_y = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_z = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.fc = nn.Sequential(nn.Linear(128, 300), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Sequential(nn.Linear(300, 21))



        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, _, L = x.shape
        out = self.conv_x(x)
        out = self.conv_y(out)
        out = self.conv_z(out)
        out = F.avg_pool1d(out, out.shape[-1]).squeeze()
        out = self.fc1(self.dropout(self.fc(out)))
        out = F.relu(out)

        return out


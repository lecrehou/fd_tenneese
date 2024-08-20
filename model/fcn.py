import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, GCNConv, SGConv, BatchNorm, GATConv  # noqa
from torch_geometric.nn import TopKPooling, EdgePooling, ASAPooling, SAGPooling, global_mean_pool



class fcn(torch.nn.Module):
    def __init__(self, feature, out_channel, pooltype, drop_rate, backbone, dropping_method, heads, K, alpha,
                 first_layer_dimension,batch_size):
        super(fcn, self).__init__()

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

        self.conv1 = nn.Conv2d(1, 128, kernel_size=(8, 1), stride=1, padding=(4, 0))
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 256, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(128, nb_classes)
        self.fc = nn.Linear(128, 21)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, drop_rate):
        x, edge_index = data.x, data.edge_index
        if 'batch' not in data:
            batch = torch.tensor([0])
            batch = batch.to(x.device)  # 设置为 1
        else:
            batch = data.batch
            batch = batch.to(x.device)
        if x.shape[1] == 500:
            x = x.narrow(1, 20, 480)
            num_nodes_per_graph = 52
            # 将 x 调整为 (batch_size, num_node_features, num_nodes)
#             x = x.reshape(-1, 1, 480, num_nodes_per_graph)
            x = x.reshape(-1, 1, 480, num_nodes_per_graph)
            x = self.relu1(self.batchnorm1(self.conv1(x)))
            x = self.relu2(self.batchnorm2(self.conv2(x)))
            x = self.relu3(self.batchnorm3(self.conv3(x)))

            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        if x.shape[1] == 960:
            x = x.narrow(1, 160, 480)
            num_nodes_per_graph = 52
            # 将 x 调整为 (batch_size, num_node_features, num_nodes)
            x = x.reshape(-1, 1, 480, num_nodes_per_graph)
#             x = x.reshape(-1, 1, 480, num_nodes_per_graph)
            x = self.relu1(self.batchnorm1(self.conv1(x)))
            x = self.relu2(self.batchnorm2(self.conv2(x)))
            x = self.relu3(self.batchnorm3(self.conv3(x)))

            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        x3 = F.log_softmax(x, dim=1)
        x4 = torch.argmax(x, dim=1)
        return x, x4,x
    #         return x3, x4

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
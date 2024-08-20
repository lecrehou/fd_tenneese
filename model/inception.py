import torch
import torch.nn as nn
from torch_geometric.nn import  BatchNorm
class InceptionModule(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=32, nb_filters=32):
        super(InceptionModule, self).__init__()

        self.conv1 = nn.Conv1d(nb_filters, nb_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(nb_filters, nb_filters, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(nb_filters, nb_filters, kernel_size=11, stride=1, padding=5)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(nb_filters, nb_filters, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(nb_filters*5)
        self.activation = nn.ReLU()

    def forward(self, x):
        channel = x.size(1)
        input_inception_model = nn.Conv1d(channel, 32, kernel_size=1, padding=0).to(x.device)
        x_input = input_inception_model(x)
        x1 = self.conv1(x_input)
        x2 = self.conv2(x_input)
        x3 = self.conv3(x_input)
        x4 = self.max_pool(x_input)
        x5 = self.conv6(x_input)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = self.batch_norm(x)
        x = self.activation(x)

        return x

class YourModel(nn.Module):
    def __init__(self, input_shape, nb_classes, depth, use_residual, nb_filters):
        super(YourModel, self).__init__()

        self.depth = depth
        self.use_residual = use_residual
        self.inception_modules = nn.ModuleList([InceptionModule(in_channels=32, nb_filters=nb_filters) for _ in range(depth)])
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters, nb_classes)
        self.softmax = nn.Softmax(dim=1)
        # 下边这里仍然要改进一下

        self.bn = BatchNorm(160)
        self.activation = nn.ReLU(inplace=True)



    def forward(self, x):
        input_layer = x
        input_res = x

        for d in range(self.depth):
            x = self.inception_modules[d](x)
            # print(d)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        conv = nn.Conv1d(input_tensor.shape[1], out_tensor.shape[1],kernel_size=1,padding=0).to(input_tensor.device)  
        shortcut_y = conv(input_tensor)
        shortcut_y = self.bn(shortcut_y)

        x = torch.add(shortcut_y, out_tensor)
        x = self.activation(x)
        return x










import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import DataParallel
adj = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0., -1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
          0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  1.,  0.]],dtype=torch.float32)

cls =torch.tensor([[0,	0,	0,	1,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
[0,	-1,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	0, -1,	0,	0, 0,	0,	0,	0],
[0,	0,	-1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	1],
[0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	-1],
[0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
[0,	0,	0,	0,	-1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0, 0,	0,	0,	0,	1],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	1,	0,	0,	0,	0,	0,	0],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0]],dtype=torch.float32)

adj2 = torch.tensor([[0,	-1,	-1,	1,	1,	1,	0,	0,	0,	0],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0,],
[1,	0,	0,	0,	0,	0,	1,	0,	0,	0,],
[-1,0,	0,	0,	0,	0,	0,	0,	0,	0,],
[0,	0,	1,	0,	0,	0,	0,	0,	0,	0,],
[0,	0,	1,	0,	-1,	0,	0,	0,	0,	1,],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,],
[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,],
[0,	0,	0,	0,	1,	0,	0,	0,	0,	0,]],dtype=torch.float32)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):

    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class unit_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(unit_gcn, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=(pad, pad), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = Variable(torch.eye(35), requires_grad=False)
        self.adj = Variable(adj, requires_grad=False)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x):
        N, C, T, V = x.size()

        f_in = x.contiguous().view(N, C * T, V)

        adj_mat = None
        self.A =self.A.cuda(x.get_device())
        
        self.adj = self.adj.cuda(x.get_device())
        
        adj_mat = self.adj[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)

        y = self.conv(torch.matmul(f_in, adj_mat_norm_d12).view(N, C, T, V))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y

class unit_gcn2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(unit_gcn2, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=(pad, pad), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = Variable(torch.eye(10), requires_grad=False)
        self.adj = Variable(adj2, requires_grad=False)
        self.cls =  Variable(cls, requires_grad=False)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x):
        cls =self.cls.cuda(x.get_device())
        
        x = torch.matmul(x,cls)
        N, C, T, V = x.size()
        f_in = x.contiguous().view(N, C * T, V)

        adj_mat = None
        self.A =self.A.cuda(x.get_device())

        self.adj = self.adj.cuda(x.get_device())
        adj_mat = self.adj[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)

        y = self.conv(torch.matmul(f_in, adj_mat_norm_d12).view(N, C, T, V))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class unit_gcn3(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(unit_gcn3, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=(pad, pad), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = Variable(torch.eye(10), requires_grad=False)
        self.adj = Variable(adj2, requires_grad=False)
        self.cls =  Variable(cls, requires_grad=False)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        f_in = x.contiguous().view(N, C * T, V)

        adj_mat = None
        self.A =self.A.cuda(x.get_device())
        self.adj = self.adj.cuda(x.get_device())
        adj_mat = self.adj[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)

        y = self.conv(torch.matmul(f_in, adj_mat_norm_d12).view(N, C, T, V))
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, kernel_size)
        self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
        # else:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


    def forward(self, x):
        x = self.gcn1(x) + self.residual(x)
        x = self.tcn1(x)
        x = self.relu(x)

        return x


class FMS(nn.Module):

    def __init__(self,k,h):
        super(FMS, self).__init__()
        self.k = k
        self.h = h
        self.average = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        N, C, T, V = x.size()
        a = T//self.k
        b = V//self.h
        d = torch.zeros((N, C, self.k, self.h),dtype=torch.float32).to(device)
        for i in range(self.k):
            for j in range(self.h):
                c = self.average(x[:,:,a*i:a*(i+1),b*j:b*(1+j)])
                d[:,:,i:i+1,j:j+1] = c
        return d




class Resnet1(nn.Module):
    def __init__(self,):
        super(Resnet1, self).__init__()
        self.l1 = unit_gcn(64,32, 1)
        self.l2 = unit_gcn(32, 64, 3)
        self.l3 = unit_gcn(64, 128, 3)
        self.res = unit_gcn(64,128, 1)
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.res(x)
        a = x3+x4
        return a


class Resnet2(nn.Module):
    def __init__(self,):
        super(Resnet2, self).__init__()
        self.l1 = unit_gcn(128, 64, 1)
        self.l2 = unit_gcn(64, 32, 3)
        self.l3 = unit_gcn2(32, 256, 3)
        self.res = unit_gcn2(128,256, 1)
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.res(x)
        a = x3+x4
        return a

class Resnet3(nn.Module):
    def __init__(self,):
        super(Resnet3, self).__init__()
        self.l1 = unit_gcn3(256, 128, 1)
        self.l2 = unit_gcn3(128, 256, 3)
        self.l3 = unit_gcn3(256, 512, 3)
        self.res = unit_gcn3(256,512, 1)
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.res(x)
        a = x3+x4
        return a

class FC1(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC1, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        bn_init(self.bn, 1)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 3))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0),1,x.size(1),1)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0),x.size(2),1)
        return x


class FC2(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC2, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm2d(out_features)
        self.soft = nn.Softmax(dim=1)
        self.tan = nn.Tanh()
        self.relu = nn.ReLU()
        bn_init(self.bn, 1)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 3))

    def forward(self, x):
        x = self.relu(x)
        x = self.fc(x)

        # x = x.view(1,7,1,1)
        # x = self.bn(x)
        # x = x.view(1,7)
        # x = self.tan(x)
        return x

class Model(nn.Module):

    def __init__(self,):
        super(Model, self).__init__()

        self.l1 = TCN_GCN_unit(1,64, 3)
        self.l2 = Resnet1()
        self.dim = TCN_GCN_unit(128,60,3)
        self.l3 = nn.AvgPool2d(kernel_size = (2,1),stride = (2,1))
        self.l4 = Resnet2()
        self.l5 = Resnet3()
        self.FMS = FMS(120,35)
#         self.fc1 = FC1(100,1)
        # 小猴修改版本
        self.fc1 = FC1(1200,1)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = FC2(512,21)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l5(x)
        # x = self.FMS(x)
        x = x.view(x.size(0),x.size(1),-1)
#         print(x.shape)
        x = self.fc1(x)
        x = self.drop(x)
        x = x.view(x.size(0),-1)
        x = self.fc2(x)

        return x

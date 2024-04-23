import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, in_channels, num_class,edge_importance_weighting,
                 temporal_kernel_size,adj_hat_path, **kwargs):
        super().__init__()
        #initialize adj
        A = np.load(adj_hat_path)
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        #initialize model parameters
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
        ))
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

        # initialize edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.Parameter(torch.ones(self.A.size())) #learnable parameter
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks) #fixed

    def forward(self, x):
        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N , V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N , C, T, V)
        #st_gcn
        for gcn in self.st_gcn_networks:
            tmp_abs_edge = torch.abs(self.edge_importance)
            x = gcn(x, self.A * ((tmp_abs_edge/2+torch.transpose(tmp_abs_edge,1,2)/2)))
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        # prediction
        x = self.fcn(x)
        x = self.sig(x)
        x = x.view(x.size(0), -1)
        return x

class st_gcn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,
                 stride=1,dropout=0.5,residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GCN(in_channels, out_channels,kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dr = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, _ = self.gcn(x, A)
        x = self.tcn(x) + res
        x = self.bn(x)
        x = self.relu(x)
        return x
        
# The based unit of graph convolutional networks.
class GCN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,
                 t_kernel_size=1,t_stride=1,t_padding=0,t_dilation=1,bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

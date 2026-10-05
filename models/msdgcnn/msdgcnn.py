"""
Multi-Scale Dynamic Graph Convolution Network for Point Clouds Classification

Paper: IEEE Access 2020
Authors: Zhengli Zhai, Xin Zhang, Luyao Yao
Source: Implementation adapted from: -
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_ops._ext as _ext

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

def farthest_point_sample(xyz, npoint):
    return _ext.furthest_point_sampling(xyz, npoint)

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

@MODELS.register_module()
class MSDGCNN(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        super(MSDGCNN, self).__init__(config, num_classes)

        self.k1 = config.k1
        self.k2 = config.k2
        self.k3 = config.k3
        self.fps_points = config.get('fps_points', 0)  # 0 = no FPS
        dropout = config.dropout

        self.bn1_1 = nn.BatchNorm2d(64)
        self.edge_conv1_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1_1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.edge_conv2_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn2_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn2_2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.edge_conv3_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn3_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn3_2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv3_3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, bias=False),
            self.bn3_3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn_agg1 = nn.BatchNorm1d(512)
        self.bn_agg2 = nn.BatchNorm1d(1024)

        self.conv_agg1 = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn_agg1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_agg2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn_agg2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn_cls1 = nn.BatchNorm1d(512)
        self.bn_cls2 = nn.BatchNorm1d(256)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.dp1 = nn.Dropout(p=dropout)

        self.linear2 = nn.Linear(512, 256, bias=False)
        self.dp2 = nn.Dropout(p=dropout)

        self.linear_final = nn.Linear(256, self.num_classes)

    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        if self.fps_points > 0 and self.fps_points < x.size(2):
            xyz = x.transpose(2, 1).contiguous()
            fps_idx = farthest_point_sample(xyz, self.fps_points)
            sampled_points = index_points(xyz, fps_idx).transpose(2, 1).contiguous()
        else:
            sampled_points = x

        x1 = get_graph_feature(sampled_points, k=self.k1)
        x1 = self.edge_conv1_1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = get_graph_feature(sampled_points, k=self.k2)
        x2 = self.edge_conv2_1(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x2 = get_graph_feature(x2, k=self.k2)
        x2 = self.edge_conv2_2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = get_graph_feature(sampled_points, k=self.k3)
        x3 = self.edge_conv3_1(x3)
        x3_1 = x3.max(dim=-1, keepdim=False)[0]

        x3 = get_graph_feature(x3_1, k=self.k3)
        x3 = self.edge_conv3_2(x3)
        x3_2 = x3.max(dim=-1, keepdim=False)[0]

        x3_shortcut = torch.cat([x3_1, x3_2], dim=1)

        x3 = get_graph_feature(x3_shortcut, k=self.k3)
        x3 = self.edge_conv3_3(x3)
        x3_final = x3.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3_final), dim=1)

        x = self.conv_agg1(x)
        x = self.conv_agg2(x)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn_cls1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)

        x = F.leaky_relu(self.bn_cls2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        x = self.linear_final(x)

        return x

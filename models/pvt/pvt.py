"""
PVT: Point-voxel Transformer for Point Cloud Learning

Paper: International Journal of Intelligent Systems 2022
Authors: Cheng Zhang, Haocheng Wan, Xinyi Shen, Zizhao Wu
Source: Implementation adapted from: https://github.com/HaochengWan/PVT
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptv_modules import SharedMLP
from .utils_pvt import create_pointnet_components
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PVT(BasePointCloudModel):
    blocks = ((64, 1, 30), (128, 2, 15), (512, 1, None), (1024, 1, None))

    def __init__(self, config, num_classes=7, channels=3, width_multiplier=1, voxel_resolution_multiplier=1, **kwargs):
        num_classes = config.num_classes
        super(PVT, self).__init__(config, num_classes)

        channels = config.channels
        width_multiplier = config.width_multiplier
        voxel_resolution_multiplier = config.voxel_resolution_multiplier
        dropout1 = config.dropout1
        dropout2 = config.dropout2

        self.in_channels = channels

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier, model='PVTConv'
        )
        self.point_features = nn.ModuleList(layers)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(channels_point + concat_channels_point + channels_point, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024))

        self.linear1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(dropout1)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(dropout2)
        self.linear3 = nn.Linear(256, self.num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, features):
        if features.shape[1] != 3 and features.shape[2] == 3:
            features = features.permute(0, 2, 1)

        num_points, batch_size = features.size(-1), features.size(0)

        coords = features[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)

        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        out_features_list.append(
            features.mean(dim=-1, keepdim=True).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_points))

        features = torch.cat(out_features_list, dim=1)
        features = F.leaky_relu(self.conv_fuse(features))
        features = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        features = F.leaky_relu(self.bn1(self.linear1(features)))
        features = self.dp1(features)
        features = F.leaky_relu(self.bn2(self.linear2(features)))
        features = self.dp2(features)
        return self.linear3(features)

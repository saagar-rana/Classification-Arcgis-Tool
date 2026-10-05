"""
PointConv: Deep Convolutional Networks on 3D Point Clouds

Paper: CVPR 2019
Authors: Wenxuan Wu, Zhongang Qi, Li Fuxin
Source: Implementation adapted from: https://github.com/DylanWusee/pointconv_pytorch
License: MIT
"""

import torch
from torch import nn
import torch.nn.functional as F

from .utils_pointconv import PointConvDensitySetAbstraction
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PointConv(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        super(PointConv, self).__init__(config, num_classes)

        channels = config.channels_input
        dropout = config.dropout
        npoint1 = config.npoint1
        npoint2 = config.npoint2
        nsample1 = config.nsample1
        nsample2 = config.nsample2
        bandwidth1 = config.bandwidth1
        bandwidth2 = config.bandwidth2
        bandwidth3 = config.bandwidth3

        self.channels = channels

        self.sa1 = PointConvDensitySetAbstraction(npoint=npoint1, nsample=nsample1, in_channel=channels, mlp=[64, 64, 128], bandwidth=bandwidth1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=npoint2, nsample=nsample2, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth=bandwidth2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth=bandwidth3, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, input_data):
        if input_data.dim() == 3 and input_data.shape[-1] == 3:
            input_data = input_data.permute(0, 2, 1)

        B, _, _ = input_data.shape
        l1_xyz, l1_points = self.sa1(input_data[:, :3, :], input_data[:, 3:, :])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x

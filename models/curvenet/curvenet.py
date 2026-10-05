"""
Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis

Paper: ICCV 2021
Authors: Tiange Xiang, Chaoyi Zhang, Yang Song, Jianhui Yu, Weidong Cai
Source: Implementation adapted from: https://github.com/tiangexiang/CurveNet
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from .curvenet_util import LPFA, CIC


@MODELS.register_module()
class CurveNet(BasePointCloudModel):

    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        super(CurveNet, self).__init__(config, num_classes)

        k = config.k
        additional_channel = config.additional_channel
        dropout = config.dropout

        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel,
                         output_channels=64, bottleneck_ratio=2, mlp_num=1,
                         curve_config=[100, 5])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64,
                         output_channels=64, bottleneck_ratio=4, mlp_num=1,
                         curve_config=[100, 5])

        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64,
                         output_channels=128, bottleneck_ratio=2, mlp_num=1,
                         curve_config=[100, 5])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128,
                         output_channels=128, bottleneck_ratio=4, mlp_num=1,
                         curve_config=[100, 5])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128,
                         output_channels=256, bottleneck_ratio=2, mlp_num=1,
                         curve_config=None)
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256,
                         output_channels=256, bottleneck_ratio=4, mlp_num=1,
                         curve_config=None)

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256,
                         output_channels=512, bottleneck_ratio=2, mlp_num=1,
                         curve_config=None)
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512,
                         output_channels=512, bottleneck_ratio=4, mlp_num=1,
                         curve_config=None)

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, self.num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)

    def forward(self, xyz):
        if xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1)

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)

        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)

        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)

        return x

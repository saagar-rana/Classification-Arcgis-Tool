"""
PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing

Paper: CVPR 2019
Authors: Hengshuang Zhao, Li Jiang, Chi-Wing Fu, Jiaya Jia
Source: Implementation adapted from: https://github.com/hszhao/PointWeb
License: MIT
"""

import torch
import torch.nn as nn

from .pointweb_module import PointWebSAModule
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PointWeb(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=0, use_xyz=True, **kwargs):
        num_classes = config.num_classes
        super(PointWeb, self).__init__(config, num_classes)

        channels = config.channels
        use_xyz = config.use_xyz
        npoint1 = config.npoint1
        npoint2 = config.npoint2
        npoint3 = config.npoint3
        npoint4 = config.npoint4
        nsample = config.nsample
        dropout1 = config.dropout1
        dropout2 = config.dropout2

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointWebSAModule(npoint=npoint1, nsample=nsample, mlp=[channels, 32, 32, 64], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=npoint2, nsample=nsample, mlp=[64, 64, 64, 128], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=npoint3, nsample=nsample, mlp=[128, 128, 128, 256], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=npoint4, nsample=nsample, mlp=[256, 256, 256, 512], mlp2=[32, 32], use_xyz=use_xyz))

        self.FC_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout2),
            nn.Linear(256, self.num_classes)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud):
        if pointcloud.shape[1] == 3 or (pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]):
            pointcloud = pointcloud.permute(0, 2, 1)

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        global_features = torch.max(features, dim=2)[0]

        return self.FC_layer(global_features)

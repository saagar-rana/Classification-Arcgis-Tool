"""
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

Paper: NeurIPS 2017
Authors: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas
Source: Implementation adapted from: https://github.com/erikwijmans/Pointnet2_PyTorch
License: Public Domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PointNet2_SSG(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointNet2_SSG, self).__init__(config, num_classes)

        self.npoint1 = config.npoint1
        self.npoint2 = config.npoint2
        self.radius1 = config.radius1
        self.radius2 = config.radius2
        self.nsample1 = config.nsample1
        self.nsample2 = config.nsample2
        self.mlp1 = config.mlp1
        self.mlp2 = config.mlp2
        self.mlp3 = config.mlp3
        self.dropout = config.dropout

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=self.npoint1,
                radius=self.radius1,
                nsample=self.nsample1,
                mlp=self.mlp1,
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=self.npoint2,
                radius=self.radius2,
                nsample=self.nsample2,
                mlp=self.mlp2,
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=self.mlp3, use_xyz=True
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        if pointcloud.shape[1] == 3 or (pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]):
            pointcloud = pointcloud.permute(0, 2, 1)

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

@MODELS.register_module()
class PointNet2_MSG(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointNet2_MSG, self).__init__(config, num_classes)

        self.npoint1 = config.npoint1
        self.npoint2 = config.npoint2
        self.radii1 = config.radii1
        self.radii2 = config.radii2
        self.nsamples1 = config.nsamples1
        self.nsamples2 = config.nsamples2
        self.mlps1 = config.mlps1
        self.mlps2 = config.mlps2
        self.mlp3 = config.mlp3
        self.dropout = config.dropout

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.npoint1,
                radii=self.radii1,
                nsamples=self.nsamples1,
                mlps=self.mlps1,
                use_xyz=True
            )
        )

        input_channels = sum([mlp[-1] for mlp in self.mlps1])
        mlps2_updated = [[input_channels] + mlp[1:] for mlp in self.mlps2]

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.npoint2,
                radii=self.radii2,
                nsamples=self.nsamples2,
                mlps=mlps2_updated,
                use_xyz=True,
            )
        )

        final_input_channels = sum([mlp[-1] for mlp in mlps2_updated])
        mlp3_updated = [final_input_channels] + self.mlp3[1:]

        self.SA_modules.append(
            PointnetSAModule(
                mlp=mlp3_updated,
                use_xyz=True,
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        if pointcloud.shape[1] == 3 or (pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]):
            pointcloud = pointcloud.permute(0, 2, 1)

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

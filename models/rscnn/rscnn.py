"""
Relation-Shape Convolutional Neural Network for Point Cloud Analysis

Paper: CVPR 2019
Authors: Yongcheng Liu, Bin Fan, Shiming Xiang, Chunhong Pan
Source: Implementation adapted from: https://github.com/Yochengliu/Relation-Shape-CNN
License: MIT
"""

import torch
import torch.nn as nn

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from .pytorch_utils import pytorch_utils as pt_utils
from .utils import PointnetSAModule, PointnetSAModuleMSG

@MODELS.register_module()
class RSCNN(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.num_classes
        super(RSCNN, self).__init__(config, num_classes)

        self.input_channels = config.input_channels
        self.relation_prior = config.relation_prior
        self.use_xyz = config.use_xyz
        self.dropout1 = config.dropout1
        self.dropout2 = config.dropout2

        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.23],
                nsamples=[48],
                mlps=[[self.input_channels, 128]],
                first_layer=True,
                use_xyz=self.use_xyz,
                relation_prior=self.relation_prior
            )
        )
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[128, 512]],
                use_xyz=self.use_xyz,
                relation_prior=self.relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                nsample=128,
                mlp=[512, 1024],
                use_xyz=self.use_xyz
            )
        )
        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=self.dropout1),
            pt_utils.FC(512, 256, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=self.dropout2),
            pt_utils.FC(256, self.num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud):
        if pointcloud.dim() == 3 and pointcloud.shape[1] == 3:
            pointcloud = pointcloud.permute(0, 2, 1)
        elif pointcloud.dim() == 3 and pointcloud.shape[1] > 3 and pointcloud.shape[1] < pointcloud.shape[2]:
            pointcloud = pointcloud.permute(0, 2, 1)

        if pointcloud.shape[-1] == 3 and self.input_channels == 3:
            pointcloud = torch.cat([pointcloud, pointcloud], dim=-1)

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.FC_layer(features.squeeze(-1))

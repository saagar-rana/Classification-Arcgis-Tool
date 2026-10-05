"""
PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing

Paper: CVPR 2019
Authors: Hengshuang Zhao, Li Jiang, Chi-Wing Fu, Jiaya Jia
Source: Implementation adapted from: https://github.com/hszhao/PointWeb
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from .pointweb_module import PointWebSAModule
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PointWeb_Seg(BaseSegModel):
    """PointWeb segmentation — SA encoder + FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        # channels = extra input features beyond xyz (0 = xyz only).
        # The SA module appends xyz (3 dims) internally when use_xyz=True,
        # so mlp[0] = channels + 3 after the += 3 inside PointWebSAModule.
        channels = config.get('channels', 0)
        use_xyz = config.get('use_xyz', True)
        nsample = config.get('nsample', 32)
        dropout = config.get('dropout', 0.5)

        # Skip channels for shallowest FP: only non-zero when extra features exist.
        # When xyz-only (channels=0), l_features[0]=None so FP receives no skip.
        skip_ch = channels

        # Encoder (4 PointWebSA stages)
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointWebSAModule(npoint=1024, nsample=nsample,
                               mlp=[channels, 32, 32, 64], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=256, nsample=nsample,
                               mlp=[64, 64, 64, 128], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=64, nsample=nsample,
                               mlp=[128, 128, 128, 256], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=16, nsample=nsample,
                               mlp=[256, 256, 256, 512], mlp2=[32, 32], use_xyz=use_xyz))

        # Decoder (4 FP stages — matching original PointWeb seg)
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + skip_ch, 128, 128, 128]))  # shallowest
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))             # deepest

        # Seg head
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()

        xyz, features = self._break_up_pc(pts)

        # Encoder
        l_xyz, l_features = [xyz], [features]
        for sa in self.SA_modules:
            li_xyz, li_features = sa(l_xyz[-1], l_features[-1])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Decoder (reverse order)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        logits = self.seg_head(l_features[0])
        return F.log_softmax(logits, dim=1)

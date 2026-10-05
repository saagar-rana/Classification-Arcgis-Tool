"""
PointConv: Deep Convolutional Networks on 3D Point Clouds

Paper: CVPR 2019
Authors: Wenxuan Wu, Zhongang Qi, Li Fuxin
Source: Implementation adapted from: https://github.com/DylanWusee/pointconv_pytorch
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from .utils_pointconv.pointconv_util import PointConvDensitySetAbstraction
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PointConv_Seg(BaseSegModel):
    """PointConv segmentation — density SA encoder + FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        # channels: raw point-cloud channel count (3=xyz only, 6=xyz+normals).
        # Extra features beyond xyz are used as the level-0 skip in fp1.
        channels = config.get('channels', 3)
        extra_ch = max(0, channels - 3)  # extra features beyond xyz
        dropout = config.get('dropout', 0.5)
        nsample1 = config.get('nsample1', 32)
        nsample2 = config.get('nsample2', 32)
        nsample3 = config.get('nsample3', 32)
        bandwidth1 = config.get('bandwidth1', 0.1)
        bandwidth2 = config.get('bandwidth2', 0.2)
        bandwidth3 = config.get('bandwidth3', 0.4)
        bandwidth4 = config.get('bandwidth4', 0.8)

        self.channels = channels

        # Encoder (4 PointConv SA stages).
        # sa1 in_channel = 3 (xyz) + extra_ch, since the grouper concatenates
        # relative xyz with any extra features.
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=1024, nsample=nsample1, in_channel=3 + extra_ch,
            mlp=[32, 32, 64], bandwidth=bandwidth1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=256, nsample=nsample2, in_channel=64 + 3,
            mlp=[64, 64, 128], bandwidth=bandwidth2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=64, nsample=nsample3, in_channel=128 + 3,
            mlp=[128, 128, 256], bandwidth=bandwidth3, group_all=False)
        self.sa4 = PointConvDensitySetAbstraction(
            npoint=16, nsample=nsample3, in_channel=256 + 3,
            mlp=[256, 256, 512], bandwidth=bandwidth4, group_all=False)

        # Decoder (4 FP stages)
        self.fp4 = PointnetFPModule(mlp=[512 + 256, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[256 + 128, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 64, 256, 128])
        # fp1 skip: only extra features concat (l0_points). xyz is NOT a skip here.
        self.fp1 = PointnetFPModule(mlp=[128 + extra_ch, 128, 128, 128])

        # Seg head
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1).contiguous()

        B, C, N = pts.shape
        l0_xyz = pts[:, :3, :]
        l0_points = pts[:, 3:, :] if C > 3 else None

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder (xyz needs [B, N, 3] for FP module)
        l0_xyz_t = l0_xyz.permute(0, 2, 1).contiguous()
        l1_xyz_t = l1_xyz.permute(0, 2, 1).contiguous()
        l2_xyz_t = l2_xyz.permute(0, 2, 1).contiguous()
        l3_xyz_t = l3_xyz.permute(0, 2, 1).contiguous()
        l4_xyz_t = l4_xyz.permute(0, 2, 1).contiguous()

        l3_points = self.fp4(l3_xyz_t, l4_xyz_t, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz_t, l3_xyz_t, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz_t, l2_xyz_t, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz_t, l1_xyz_t, l0_points, l1_points)

        logits = self.seg_head(l0_points)
        return F.log_softmax(logits, dim=1)

"""
PCT: Point Cloud Transformer

Paper: Computational Visual Media 2021
Authors: Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu
Source: Implementation adapted from: https://github.com/MenghaoGuo/PCT
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from .pct_layers import Point_Transformer_Last, Local_op
from .pct_utils import sample_and_group
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PCT_Seg(BaseSegModel):
    """PCT segmentation — offset-attention encoder + FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        dropout = config.get('dropout', 0.5)
        npoint1 = config.get('npoint1', 512)
        npoint2 = config.get('npoint2', 256)
        radius1 = config.get('radius1', 0.15)
        radius2 = config.get('radius2', 0.2)
        nsample = config.get('nsample', 32)

        self.npoint1 = npoint1
        self.npoint2 = npoint2
        self.radius1 = radius1
        self.radius2 = radius2
        self.nsample = nsample

        # Initial per-point feature extraction
        self.conv1 = nn.Conv1d(channels, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Encoder: sample_and_group + local_op
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)  # 64*2=128 → 128
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)  # 128*2=256 → 256

        # 4 offset-attention layers
        self.pt_last = Point_Transformer_Last()

        # Fuse: SA output(256*4=1024) + feature_1(256) = 1280 → 1024
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        # Decoder: FP upsampling with skip connections
        self.fp2 = PointnetFPModule(mlp=[1024 + 128, 256, 256])   # 256pts→512pts
        self.fp1 = PointnetFPModule(mlp=[256 + 64, 128, 128])     # 512pts→Npts

        # Seg head
        label_dim = 0
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + label_dim, 128, 1, bias=False),
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
        xyz = pts.permute(0, 2, 1).contiguous()  # [B, N, 3]

        # Initial per-point features
        x = F.relu(self.bn1(self.conv1(pts)))     # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))       # [B, 64, N]

        l0_xyz = xyz                               # [B, N, 3]
        l0_features = x                            # [B, 64, N]

        # Stage 1: sample_and_group N→npoint1
        x_bnc = x.permute(0, 2, 1)                # [B, N, 64]
        new_xyz1, new_feature1 = sample_and_group(
            npoint=self.npoint1, radius=self.radius1, nsample=self.nsample,
            xyz=xyz, points=x_bnc)
        feature_0 = self.gather_local_0(new_feature1)  # [B, 128, npoint1]

        l1_xyz = new_xyz1                          # [B, npoint1, 3]
        l1_features = feature_0                    # [B, 128, npoint1]

        # Stage 2: sample_and_group npoint1→npoint2
        feature_bnc = feature_0.permute(0, 2, 1)   # [B, npoint1, 128]
        new_xyz2, new_feature2 = sample_and_group(
            npoint=self.npoint2, radius=self.radius2, nsample=self.nsample,
            xyz=new_xyz1, points=feature_bnc)
        feature_1 = self.gather_local_1(new_feature2)  # [B, 256, npoint2]

        l2_xyz = new_xyz2                           # [B, npoint2, 3]

        # 4 offset-attention layers on npoint2 points
        x = self.pt_last(feature_1)                 # [B, 1024, npoint2]
        x = torch.cat([x, feature_1], dim=1)        # [B, 1280, npoint2]
        x = self.conv_fuse(x)                        # [B, 1024, npoint2]

        # Decoder: upsample back to N
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_features, x)     # [B, 256, npoint1]
        l0_up = self.fp1(l0_xyz, l1_xyz, l0_features, l1_up)  # [B, 128, N]

        # Seg head
        parts = [l0_up]
        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1)).expand(-1, -1, N)
            parts.append(cls_feat)

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)

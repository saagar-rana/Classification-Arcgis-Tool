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

from .curvenet_util import CIC, LPFA, PointNetFeaturePropagation
from ..build import MODELS
from ..base_seg_model import BaseSegModel


class CurveNetSegBase(nn.Module):
    """CurveNet encoder-decoder backbone for segmentation.

    Curve aggregation (`[num_curves, curve_length] = [100, 5]`) is used in
    the first two encoder stages; disabled in deeper stages — the paper's
    "default" configuration.
    """

    def __init__(self, num_classes=50, category=16, k=32):
        super().__init__()

        additional_channel = 32
        self.category = category

        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # Encoder
        self.cic11 = CIC(npoint=2048, radius=0.2, k=k, in_channels=additional_channel,
                         output_channels=64, bottleneck_ratio=2, curve_config=[100, 5])
        self.cic12 = CIC(npoint=2048, radius=0.2, k=k, in_channels=64,
                         output_channels=64, bottleneck_ratio=4, curve_config=[100, 5])

        self.cic21 = CIC(npoint=512, radius=0.4, k=k, in_channels=64,
                         output_channels=128, bottleneck_ratio=2, curve_config=[100, 5])
        self.cic22 = CIC(npoint=512, radius=0.4, k=k, in_channels=128,
                         output_channels=128, bottleneck_ratio=4, curve_config=[100, 5])

        self.cic31 = CIC(npoint=128, radius=0.8, k=k, in_channels=128,
                         output_channels=256, bottleneck_ratio=2, curve_config=None)
        self.cic32 = CIC(npoint=128, radius=0.8, k=k, in_channels=256,
                         output_channels=256, bottleneck_ratio=4, curve_config=None)

        self.cic41 = CIC(npoint=32, radius=1.2, k=31, in_channels=256,
                         output_channels=512, bottleneck_ratio=2, curve_config=None)
        self.cic42 = CIC(npoint=32, radius=1.2, k=31, in_channels=512,
                         output_channels=512, bottleneck_ratio=4, curve_config=None)

        self.cic51 = CIC(npoint=8, radius=2.0, k=7, in_channels=512,
                         output_channels=1024, bottleneck_ratio=2, curve_config=None)
        self.cic52 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024,
                         output_channels=1024, bottleneck_ratio=4, curve_config=None)
        self.cic53 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024,
                         output_channels=1024, bottleneck_ratio=4, curve_config=None)

        # Decoder
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512],
                                              att=[1024, 512, 256])
        self.up_cic5 = CIC(npoint=32, radius=1.2, k=31, in_channels=512,
                           output_channels=512, bottleneck_ratio=4)

        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256],
                                              att=[512, 256, 128])
        self.up_cic4 = CIC(npoint=128, radius=0.8, k=k, in_channels=256,
                           output_channels=256, bottleneck_ratio=4)

        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128],
                                              att=[256, 128, 64])
        self.up_cic3 = CIC(npoint=512, radius=0.4, k=k, in_channels=128,
                           output_channels=128, bottleneck_ratio=4)

        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[64, 64],
                                              att=[128, 64, 32])

        # Final CIC layers after concatenation with global features + class label
        # l1_xyz(3) + l1_points(64) + emb1(64) + emb2(128) + category
        up_cic2_in = 3 + 64 + 64 + 128 + category
        self.up_cic2 = CIC(npoint=2048, radius=0.2, k=k,
                           in_channels=up_cic2_in,
                           output_channels=256, bottleneck_ratio=4)
        self.up_cic1 = CIC(npoint=2048, radius=0.2, k=k, in_channels=256,
                           output_channels=256, bottleneck_ratio=4)

        # Global feature projections
        self.global_conv2 = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.global_conv1 = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Seg head with SE attention
        self.conv1 = nn.Conv1d(256, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, num_classes, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(256, 256 // 8, 1, bias=False),
            nn.BatchNorm1d(256 // 8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256 // 8, 256, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, xyz, cls_label=None):
        batch_size = xyz.size(0)

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        l5_xyz, l5_points = self.cic51(l4_xyz, l4_points)
        l5_xyz, l5_points = self.cic52(l5_xyz, l5_points)
        l5_xyz, l5_points = self.cic53(l5_xyz, l5_points)

        # Global features
        emb1 = self.global_conv1(l4_points).max(dim=-1, keepdim=True)[0]  # [B, 64, 1]
        emb2 = self.global_conv2(l5_points).max(dim=-1, keepdim=True)[0]  # [B, 128, 1]

        # Decoder
        l4_points = self.fp4(l4_xyz, l5_xyz, l4_points, l5_points)
        l4_xyz, l4_points = self.up_cic5(l4_xyz, l4_points)

        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_xyz, l3_points = self.up_cic4(l3_xyz, l3_points)

        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_xyz, l2_points = self.up_cic3(l2_xyz, l2_points)

        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        # Concatenate local features + global features + class label
        if cls_label is not None:
            cls_label = cls_label.view(batch_size, -1, 1)  # [B, C, 1]
            emb = torch.cat((emb1, emb2, cls_label), dim=1)
        else:
            emb = torch.cat((emb1, emb2), dim=1)
        emb = emb.expand(-1, -1, xyz.size(-1))

        x = torch.cat((l1_xyz, l1_points, emb), dim=1)
        xyz, x = self.up_cic2(l1_xyz, x)
        xyz, x = self.up_cic1(xyz, x)

        # Seg head with SE
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        se = self.se(x)
        x = x * se
        x = self.drop1(x)
        x = self.conv2(x)
        return x  # [B, num_classes, N]


@MODELS.register_module()
class CurveNet_Seg(BaseSegModel):
    """CurveNet segmentation model for LIDARLearn."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        k = config.get('k', 32)

        # Adjust up_cic2 input channels based on whether cls_label is used
        category = self.num_obj_classes if self.use_cls_label else 0

        self.backbone = CurveNetSegBase(
            num_classes=self.seg_classes,
            category=category,
            k=k,
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, C, N] — CurveNet expects [B, 3, N]
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1).contiguous()

        label = cls_label if self.use_cls_label else None
        logits = self.backbone(pts, label)  # [B, seg_classes, N]
        return F.log_softmax(logits, dim=1)

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

from .utils_pvt import create_pointnet_components, create_mlp_components
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PVT_Seg(BaseSegModel):
    """PVT part segmentation — all N points preserved via PartPVTConv."""

    blocks = ((64, 1, 30), (128, 2, 15), (512, 1, None), (1024, 1, None))

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 6)
        width_multiplier = config.get('width_multiplier', 1)
        voxel_resolution_multiplier = config.get('voxel_resolution_multiplier', 1)

        self.in_channels = channels

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, normalize=False,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            model='PartPVTConv',
        )
        self.point_features = nn.ModuleList(layers)

        # Classifier input: one_hot + all intermediate + global_max + global_avg
        label_dim = self.num_obj_classes if self.use_cls_label else 0
        in_cls = label_dim + channels_point + concat_channels_point + channels_point
        layers, _ = create_mlp_components(
            in_channels=in_cls,
            out_channels=[256, 0.2, 256, 0.2, 128, self.seg_classes],
            classifier=True, dim=2, width_multiplier=width_multiplier,
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, pts, cls_label=None):
        # pts: [B, C, N] or [B, N, C]
        if pts.shape[1] != self.in_channels and pts.shape[2] <= self.in_channels:
            pts = pts.permute(0, 2, 1).contiguous()

        B = pts.size(0)
        N = pts.size(-1)
        features = pts
        coords = features[:, :3, :]

        out_list = []

        # One-hot class label broadcast
        if self.use_cls_label and cls_label is not None:
            out_list.append(cls_label.unsqueeze(-1).expand(-1, -1, N))

        # Run PartPVTConv blocks
        for pf in self.point_features:
            features, _ = pf((features, coords))
            out_list.append(features)

        # Global features
        out_list.append(features.max(dim=-1, keepdim=True).values.expand(-1, -1, N))
        out_list.append(features.mean(dim=-1, keepdim=True).expand(-1, -1, N))

        logits = self.classifier(torch.cat(out_list, dim=1))  # [B, seg_classes, N]
        return F.log_softmax(logits, dim=1)

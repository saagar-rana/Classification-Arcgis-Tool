"""
PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets

Paper: Computers & Graphics 2025 (arXiv 2410.10084)
Authors: Ali Kashefi
Source: Implementation adapted from: -
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointkan import KANshared
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PointKAN_Seg(BaseSegModel):
    """PointKAN segmentation — per-point KAN features + global pooling."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        poly_degree = config.get('poly_degree', 3)
        alpha = config.get('alpha', 1.0)
        beta = config.get('beta', 1.0)
        scale = config.get('scale', 1.0)
        dropout = config.get('dropout', 0.5)

        feat_dim = int(1024 * scale)

        # Per-point KAN encoder (preserves N points)
        self.jacobikan = KANshared(channels, feat_dim, poly_degree, alpha, beta)
        self.bn = nn.BatchNorm1d(feat_dim)

        # Class label projection for part seg
        label_dim = 0
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        # Seg head: per_point(feat_dim) + global_max(feat_dim) + global_avg(feat_dim) + label
        in_dim = feat_dim * 3 + label_dim
        self.seg_head = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1).contiguous()

        B, C, N = pts.shape

        # Per-point features
        x = self.jacobikan(pts)   # [B, feat_dim, N]
        x = self.bn(x)

        # Global features
        g_max = x.max(dim=-1)[0]   # [B, feat_dim]
        g_avg = x.mean(dim=-1)     # [B, feat_dim]
        g = torch.cat([g_max, g_avg], dim=1).unsqueeze(-1).expand(-1, -1, N)

        # Concat per-point + global
        parts = [x, g]

        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))  # [B, 64, 1]
            parts.append(cls_feat.expand(-1, -1, N))

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)

"""
Multi-Scale Dynamic Graph Convolution Network for Point Clouds Classification

Paper: IEEE Access 2020
Authors: Zhengli Zhai, Xin Zhang, Luyao Yao
Source: Implementation adapted from: -
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .msdgcnn import get_graph_feature
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class MSDGCNN_Seg(BaseSegModel):
    """MS-DGCNN segmentation — no FPS, all N points preserved."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        k1 = config.get('k1', 5)
        k2 = config.get('k2', 20)
        k3 = config.get('k3', 30)
        dropout = config.get('dropout', 0.5)

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        # Branch 1: single scale k1
        self.edge_conv1_1 = nn.Sequential(
            nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))

        # Branch 2: two layers at k2
        self.edge_conv2_1 = nn.Sequential(
            nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.edge_conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))

        # Branch 3: three layers at k3
        self.edge_conv3_1 = nn.Sequential(
            nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.edge_conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.edge_conv3_3 = nn.Sequential(
            nn.Conv2d(384, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

        # Aggregation: 64 + 128 + 256 = 448 → 512 → 1024
        self.conv_agg1 = nn.Sequential(
            nn.Conv1d(448, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.conv_agg2 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))

        # Class label for part seg
        label_dim = 0
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        # Seg head: per_point(1024) + global_max(1024) + global_avg(1024) + label
        in_dim = 1024 * 3 + label_dim
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
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1).contiguous()

        B, C, N = pts.shape

        # Branch 1
        x1 = get_graph_feature(pts, k=self.k1)
        x1 = self.edge_conv1_1(x1).max(dim=-1)[0]  # [B, 64, N]

        # Branch 2
        x2 = get_graph_feature(pts, k=self.k2)
        x2 = self.edge_conv2_1(x2).max(dim=-1)[0]  # [B, 64, N]
        x2 = get_graph_feature(x2, k=self.k2)
        x2 = self.edge_conv2_2(x2).max(dim=-1)[0]  # [B, 128, N]

        # Branch 3
        x3 = get_graph_feature(pts, k=self.k3)
        x3 = self.edge_conv3_1(x3).max(dim=-1)[0]  # [B, 64, N]
        x3_1 = x3

        x3 = get_graph_feature(x3_1, k=self.k3)
        x3 = self.edge_conv3_2(x3).max(dim=-1)[0]  # [B, 128, N]
        x3_2 = x3

        x3_shortcut = torch.cat([x3_1, x3_2], dim=1)  # [B, 192, N]
        x3 = get_graph_feature(x3_shortcut, k=self.k3)
        x3 = self.edge_conv3_3(x3).max(dim=-1)[0]  # [B, 256, N]

        # Concat + aggregate
        x = torch.cat((x1, x2, x3), dim=1)  # [B, 448, N]
        x = self.conv_agg1(x)   # [B, 512, N]
        x = self.conv_agg2(x)   # [B, 1024, N]

        # Global features
        g_max = x.max(dim=-1)[0].unsqueeze(-1).expand(-1, -1, N)
        g_avg = x.mean(dim=-1).unsqueeze(-1).expand(-1, -1, N)

        parts = [x, g_max, g_avg]
        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1)).expand(-1, -1, N)
            parts.append(cls_feat)

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)

"""
P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting

Paper: NeurIPS 2022
Authors: Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, Jiwen Lu
Source: Implementation adapted from: https://github.com/wangzy22/P2P
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import knn_point
from ..build import MODELS
from ..base_seg_model import BaseSegModel


class GraphFeatureEncoder(nn.Module):
    """Extract per-point features via local graph convolution (from P2P encoder)."""

    def __init__(self, trans_dim=64, graph_dim=128, local_size=20):
        super().__init__()
        self.local_size = local_size
        self.graph_dim = graph_dim

        self.input_trans = nn.Conv1d(3, trans_dim, 1)
        self.graph_layer = nn.Sequential(
            nn.Conv2d(trans_dim * 2, graph_dim, kernel_size=1, bias=False),
            nn.GroupNorm(4, graph_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.proj_layer = nn.Conv1d(graph_dim, graph_dim, kernel_size=1)

    @staticmethod
    def get_graph_feature(coor, x, k):
        B, C, N = x.shape
        with torch.no_grad():
            idx = knn_point(k, coor.transpose(1, 2).contiguous(),
                            coor.transpose(1, 2).contiguous())
            idx = idx.transpose(1, 2).contiguous()  # [B, k, N]
            idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
            idx_flat = (idx + idx_base).view(-1)

        x_t = x.transpose(2, 1).contiguous()  # [B, N, C]
        feature = x_t.view(B * N, -1)[idx_flat, :]
        feature = feature.view(B, k, N, C).permute(0, 3, 2, 1).contiguous()
        x_exp = x.unsqueeze(-1).expand(-1, -1, -1, k)
        return torch.cat((feature - x_exp, x_exp), dim=1)  # [B, 2C, N, k]

    def forward(self, xyz):
        """
        xyz: [B, 3, N]
        Returns: [B, graph_dim, N] per-point features
        """
        f = self.input_trans(xyz)  # [B, trans_dim, N]
        f = self.get_graph_feature(xyz, f, self.local_size)  # [B, 2*trans_dim, N, k]
        f = self.graph_layer(f)  # [B, graph_dim, N, k]
        f = f.max(dim=-1)[0]  # [B, graph_dim, N]
        f = self.proj_layer(f)  # [B, graph_dim, N]
        return f


@MODELS.register_module()
class P2P_Seg(BaseSegModel):
    """P2P segmentation using per-point graph features."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        trans_dim = config.get('trans_dim', 64)
        graph_dim = config.get('graph_dim', 128)
        local_size = config.get('local_size', 20)
        dropout = config.get('dropout', 0.5)

        self.encoder = GraphFeatureEncoder(
            trans_dim=trans_dim, graph_dim=graph_dim, local_size=local_size)

        # Label projection for part seg
        label_dim = 0
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        # Seg head: per_point(graph_dim) + global_max(graph_dim) + global_avg(graph_dim) + label
        in_dim = graph_dim * 3 + label_dim
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

        # Per-point graph features
        f = self.encoder(pts[:, :3, :])  # [B, graph_dim, N]

        # Global features
        g_max = f.max(dim=-1)[0].unsqueeze(-1).expand(-1, -1, N)
        g_avg = f.mean(dim=-1).unsqueeze(-1).expand(-1, -1, N)

        parts = [f, g_max, g_avg]
        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1)).expand(-1, -1, N)
            parts.append(cls_feat)

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)

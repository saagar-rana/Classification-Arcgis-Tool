"""
Points to Patches: Enabling the Use of Self-Attention for 3D Shape Recognition

Paper: ICPR 2022
Authors: Axel Berg, Magnus Oskarsson, Mark O'Connor
Source: Implementation adapted from: https://github.com/axeber01/point-tnt
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .tnt_layers import PreNorm, FeedForward, Attention
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class GlobalTransformer_Seg(BaseSegModel):
    """GlobalTransformer segmentation — all N points, global self-attention."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        patch_dim = config.get('patch_dim', 256)
        depth = config.get('depth', 4)
        heads = config.get('heads', 8)
        dim_head = config.get('dim_head', 64)
        emb_dims = config.get('emb_dims', 512)
        attn_dropout = config.get('attn_dropout', 0.0)
        ff_dropout = config.get('ff_dropout', 0.0)
        dropout = config.get('dropout', 0.5)

        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, patch_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(patch_dim, Attention(dim=patch_dim, heads=heads,
                        dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim=patch_dim, dropout=ff_dropout)),
            ]))

        self.final_conv = nn.Sequential(
            nn.LayerNorm(patch_dim * depth),
            nn.Linear(patch_dim * depth, emb_dims),
            nn.GELU(),
        )

        # Label projection for part seg
        label_dim = 0
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        # Seg head: per_point(emb_dims) + global_max(emb_dims) + global_avg(emb_dims) + label
        in_dim = emb_dims * 3 + label_dim
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

        # Per-point features via global self-attention
        points = self.to_anchor(pts)  # [B, N, patch_dim]
        ylist = []
        for attn, ff in self.layers:
            points = attn(points) + points
            points = ff(points) + points
            ylist.append(points)

        y = torch.cat(ylist, dim=-1)       # [B, N, patch_dim * depth]
        y = self.final_conv(y)             # [B, N, emb_dims]
        y = y.permute(0, 2, 1).contiguous()  # [B, emb_dims, N]

        # Global features
        g_max = y.max(dim=-1)[0].unsqueeze(-1).expand(-1, -1, N)
        g_avg = y.mean(dim=-1).unsqueeze(-1).expand(-1, -1, N)

        parts = [y, g_max, g_avg]
        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1)).expand(-1, -1, N)
            parts.append(cls_feat)

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)

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
from einops import rearrange
from einops.layers.torch import Rearrange
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from .tnt_layers import PreNorm, FeedForward, Attention
from .tnt_helpers import farthest_point_sample, index_points, get_graph_feature as get_graph_feature_tnt
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PointTNT_Seg(BaseSegModel):
    """PointTNT segmentation — patch attention + FP upsampling."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        point_dim = config.get('point_dim', 128)
        patch_dim = config.get('patch_dim', 256)
        depth = config.get('depth', 4)
        heads = config.get('heads', 8)
        dim_head = config.get('dim_head', 64)
        ff_dropout = config.get('ff_dropout', 0.0)
        attn_dropout = config.get('attn_dropout', 0.0)
        emb_dims = config.get('emb_dims', 512)
        n_anchor = config.get('n_anchor', 256)
        k = config.get('k', 20)
        dilation = config.get('dilation', 1)
        dropout = config.get('dropout', 0.5)

        self.n_anchor = n_anchor
        self.n_neigh = k
        self.dilation = dilation

        # Initial per-point feature (for skip connection)
        self.input_proj = nn.Sequential(
            nn.Conv1d(channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )

        # Patch encoder
        self.to_point_tokens = nn.Sequential(
            Rearrange('b c (p) (n) -> (b p) n c'),
            nn.Linear(channels, point_dim),
        )
        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, patch_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            point_to_patch = nn.Sequential(
                nn.LayerNorm(2 * point_dim),
                nn.Linear(2 * point_dim, patch_dim),
            )
            self.layers.append(nn.ModuleList([
                PreNorm(point_dim, Attention(dim=point_dim, heads=heads,
                        dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(point_dim, FeedForward(dim=point_dim, dropout=ff_dropout)),
                point_to_patch,
                PreNorm(patch_dim, Attention(dim=patch_dim, heads=heads,
                        dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim=patch_dim, dropout=ff_dropout)),
            ]))

        self.final_conv = nn.Sequential(
            nn.LayerNorm(patch_dim * depth),
            nn.Linear(patch_dim * depth, emb_dims),
            nn.GELU(),
        )

        # FP: upsample from n_anchor to N
        self.fp = PointnetFPModule(mlp=[emb_dims + 64, 256, 128])

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
        x = pts  # [B, 3, N]

        # Per-point skip features
        l0_feat = self.input_proj(x)  # [B, 64, N]
        l0_xyz = x[:, :3, :].permute(0, 2, 1).contiguous()  # [B, N, 3]

        # FPS anchor selection
        fps_idx = farthest_point_sample(xyz=x.permute(0, 2, 1), npoint=self.n_anchor)
        anchors = index_points(x.permute(0, 2, 1), fps_idx).permute(0, 2, 1)  # [B, 3, n_anchor]
        anchor_xyz = anchors[:, :3, :].permute(0, 2, 1).contiguous()  # [B, n_anchor, 3]

        patches = self.to_anchor(anchors)  # [B, n_anchor, patch_dim]

        # k-NN edges around each anchor
        e = get_graph_feature_tnt(x, k=self.n_neigh, d=self.dilation)
        e = e[:, 0:3, :, :]
        e = index_points(e.permute(0, 2, 1, 3), fps_idx).permute(0, 2, 1, 3)
        edges = self.to_point_tokens(e)  # [(B*n_anchor), k, point_dim]

        # Two-level transformer
        ylist = []
        for point_attn, point_ff, point_to_patch_res, patch_attn, patch_ff in self.layers:
            edges = point_attn(edges) + edges
            edges = point_ff(edges) + edges

            p1 = edges.max(dim=1)[0]
            p2 = edges.mean(dim=1)
            edge_features = torch.cat((p1, p2), dim=1)

            patches_res = point_to_patch_res(edge_features)
            patches_res = rearrange(patches_res, '(b p) d -> b p d', b=B)
            patches = patches + patches_res

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches
            ylist.append(patches)

        y = torch.cat(ylist, dim=-1)       # [B, n_anchor, patch_dim*depth]
        y = self.final_conv(y)             # [B, n_anchor, emb_dims]
        y = y.permute(0, 2, 1).contiguous()  # [B, emb_dims, n_anchor]

        # Upsample: n_anchor → N via FP
        up = self.fp(l0_xyz, anchor_xyz, l0_feat, y)  # [B, 128, N]

        # Seg head
        parts = [up]
        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1)).expand(-1, -1, N)
            parts.append(cls_feat)

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)

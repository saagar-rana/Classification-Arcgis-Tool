"""
Points to Patches: Enabling the Use of Self-Attention for 3D Shape Recognition

Paper: ICPR 2022
Authors: Axel Berg, Magnus Oskarsson, Mark O'Connor
Source: Implementation adapted from: https://github.com/axeber01/point-tnt
License: MIT
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .tnt_layers import PreNorm, FeedForward, Attention
from .tnt_helpers import farthest_point_sample, index_points, get_graph_feature as get_graph_feature_tnt
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PointTNT(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        num_classes = config.num_classes
        super().__init__(config, num_classes)

        channels = config.channels
        point_dim = config.point_dim
        patch_dim = config.patch_dim
        depth = config.depth
        heads = config.heads
        dim_head = config.dim_head
        ff_dropout = config.ff_dropout
        attn_dropout = config.attn_dropout
        emb_dims = config.emb_dims
        n_anchor = config.n_anchor
        k = config.k
        dilation = config.dilation
        dropout = config.dropout

        self.n_anchor = n_anchor
        self.n_neigh = k
        self.dilation = dilation

        self.to_point_tokens = nn.Sequential(
            Rearrange('b c (p) (n) -> (b p) n c'),
            nn.Linear(channels, point_dim)
        )

        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, patch_dim)
        )

        layers = nn.ModuleList([])
        for _ in range(depth):
            point_to_patch = nn.Sequential(
                nn.LayerNorm(2 * point_dim),
                nn.Linear(2 * point_dim, patch_dim),
            )

            layers.append(nn.ModuleList([
                PreNorm(point_dim, Attention(dim=point_dim, heads=heads, dim_head=dim_head,
                        dropout=attn_dropout)),
                PreNorm(point_dim, FeedForward(dim=point_dim, dropout=ff_dropout)),
                point_to_patch,
                PreNorm(patch_dim, Attention(dim=patch_dim, heads=heads, dim_head=dim_head,
                        dropout=attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim=patch_dim, dropout=ff_dropout)),
            ]))

        self.layers = layers

        self.final_conv = nn.Sequential(
            nn.LayerNorm(patch_dim * depth),
            nn.Linear(patch_dim * depth, emb_dims),
            nn.GELU(),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, pointcloud):
        if pointcloud.shape[1] == 3 or (pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]):
            x = pointcloud
        else:
            x = pointcloud.permute(0, 2, 1)

        batch_size, _, _ = x.shape

        fps_idx = farthest_point_sample(xyz=x.permute(0, 2, 1), npoint=self.n_anchor)
        anchors = index_points(x.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
        torch.cuda.empty_cache()

        patches = self.to_anchor(anchors)

        e = get_graph_feature_tnt(x, k=self.n_neigh, d=self.dilation)
        e = e[:, 0:3, :, :]

        e = index_points(e.permute(0, 2, 1, 3), fps_idx).permute(0, 2, 1, 3)

        edges = self.to_point_tokens(e)

        ylist = []

        for point_attn, point_ff, point_to_patch_residual, patch_attn, patch_ff in self.layers:
            edges = point_attn(edges) + edges
            edges = point_ff(edges) + edges

            p1 = edges.max(dim=1, keepdim=False)[0]
            p2 = edges.mean(dim=1, keepdim=False)
            edge_features = torch.cat((p1, p2), dim=1)

            patches_residual = point_to_patch_residual(edge_features)
            patches_residual = rearrange(patches_residual, '(b p) d -> b p d', b=batch_size)
            patches = patches + patches_residual

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches

            ylist.append(patches)

        y = torch.cat(ylist, dim=-1)
        y = self.final_conv(y)

        y1 = y.max(dim=1, keepdim=False)[0]
        y2 = y.mean(dim=1, keepdim=False)
        y = torch.cat((y1, y2), dim=1)

        return self.mlp_head(y)

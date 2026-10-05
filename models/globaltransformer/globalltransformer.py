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
from ..base_model import BasePointCloudModel, print_model_info


@MODELS.register_module()
class GlobalTransformer(BasePointCloudModel):
    def __init__(
        self,
        config,
        num_classes=7,
        channels=3,
        **kwargs,
    ):
        super(GlobalTransformer, self).__init__(config, num_classes)

        self.global_attention = config.global_attention

        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, config.patch_dim)
        )

        layers = nn.ModuleList([])
        for _ in range(config.depth):

            layers.append(nn.ModuleList([
                PreNorm(config.patch_dim, Attention(dim=config.patch_dim, heads=config.heads, dim_head=config.dim_head,
                        dropout=config.attn_dropout)) if self.global_attention else nn.Identity(),
                PreNorm(config.patch_dim, FeedForward(
                    dim=config.patch_dim, dropout=config.ff_dropout)),
            ]))

        self.layers = layers

        self.final_conv = nn.Sequential(
            nn.LayerNorm(config.patch_dim * config.depth),
            nn.Linear(config.patch_dim * config.depth, config.emb_dims),
            nn.GELU(),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(config.emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.shape
        points = self.to_anchor(x)
        ylist = []

        for patch_attn, patch_ff in self.layers:
            points = patch_attn(points) + points
            points = patch_ff(points) + points
            ylist.append(points)

        y = torch.cat(ylist, dim=-1)
        y = self.final_conv(y)

        y1 = y.max(dim=1, keepdim=False)[0]
        y2 = y.mean(dim=1, keepdim=False)
        y = torch.cat((y1, y2), dim=1)

        return self.mlp_head(y)

"""
PointGPT: Auto-regressively Generative Pre-training from Point Clouds

Paper: NeurIPS 2023
Authors: Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue
Source: Implementation adapted from: https://github.com/CGuangyan-BIT/PointGPT
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Block(nn.Module):
    """Standard GPT block - same as original for checkpoint compatibility."""

    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, attn_mask):
        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT_extractor_PPT(nn.Module):
    """
    GPT extractor with PPT prompt tuning.
    Uses the same layer structure as original GPT_extractor for checkpoint compatibility,
    with added prompt MLPs applied during forward pass.
    """

    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size,
        num_group=64, num_group_extra=64, pretrained=False
    ):
        super(GPT_extractor_PPT, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.num_group = num_group
        self.num_group_extra = num_group_extra

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # Standard GPT layers (same structure as original for checkpoint loading)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * (self.group_size), 1)
        )

        if pretrained == False:
            # PPT uses 3-way concatenation: cls + points_mean + center_mean
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 3, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

            self.cls_norm = nn.LayerNorm(self.trans_dim)

        # PPT prompt MLPs (new trainable parameters)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.prompt_pos_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.prompt_all_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        # Initialize prompt MLPs
        self._init_prompt_weights()

    def _init_prompt_weights(self):
        for module in [self.prompt_mlp, self.prompt_pos_mlp, self.prompt_all_mlp]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, h, pos, attn_mask, classify=False):
        """
        Forward pass with PPT prompt tuning.
        h: [batch, length, C] - input tokens (cls + points + centers)
        pos: [batch, length, C] - position embeddings
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)  # [length, batch, C]
        pos = pos.transpose(0, 1)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:-1, :, :]], axis=0)
        else:
            h = torch.cat([sos, h], axis=0)

        # Token layout after sos prepend: [sos, cls, points..., centers...]
        point_start = 2
        point_end = point_start + self.num_group
        center_start = h.shape[0] - self.num_group_extra

        # transformer with PPT prompts
        for layer in self.layers:
            # Add position embeddings (create new tensor, no inplace)
            h_with_pos = h + pos

            # Apply prompt MLPs using concatenation (no inplace)
            # Split into segments: [sos+cls, points, middle (if any), centers]
            seg_before = h_with_pos[:point_start]  # sos + cls
            seg_points = h_with_pos[point_start:point_end]  # point tokens
            seg_centers = h_with_pos[center_start:]  # center tokens

            # Apply prompts
            seg_points_prompted = seg_points + self.prompt_mlp(seg_points)
            seg_centers_prompted = seg_centers + self.prompt_pos_mlp(seg_centers)

            # Handle middle segment if exists (between points and centers)
            if center_start > point_end:
                seg_middle = h_with_pos[point_end:center_start]
                h_with_pos = torch.cat([seg_before, seg_points_prompted, seg_middle, seg_centers_prompted], dim=0)
            else:
                h_with_pos = torch.cat([seg_before, seg_points_prompted, seg_centers_prompted], dim=0)

            # Apply block with prompt_all_mlp residual
            h = layer(h_with_pos, attn_mask)
            h = h + self.prompt_all_mlp(h)

        h = self.ln_f(h)

        encoded_points = h.transpose(0, 1)
        if not classify:
            return encoded_points

        h = h.transpose(0, 1)  # [batch, length+1, C]
        h = self.cls_norm(h)

        # PPT 3-way concatenation: cls + points_mean + centers_mean
        cls_token = h[:, 1]  # cls token (after sos)
        point_tokens = h[:, 2:2 + self.num_group]  # point tokens
        center_tokens = h[:, -self.num_group_extra:]  # center tokens

        concat_f = torch.cat([
            cls_token,
            point_tokens.mean(1),
            center_tokens.mean(1)
        ], dim=-1)

        ret = self.cls_head_finetune(concat_f)
        return ret, encoded_points

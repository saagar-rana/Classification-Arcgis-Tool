"""
Shared building blocks for PointGPT-family models.

PointGPT architecture from:
    Chen et al., "PointGPT: Auto-regressively Generative Pre-training from
    Point Clouds", NeurIPS 2023.
    https://github.com/CGuangyan-BIT/PointGPT  (MIT License)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.knn import KNN

from utils import misc


class Encoder_large(nn.Module):
    """PointGPT encoder for encoder_dims >= 768."""

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Encoder_small(nn.Module):
    """PointGPT encoder for encoder_dims == 384."""

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class PositionEmbeddingCoordsSine(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates."""

    def __init__(self, n_dim: int = 1, d_model: int = 256,
                 temperature=10000, scale=None):
        super().__init__()
        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim
        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.shape[-1] == self.n_dim
        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode='trunc')
            / self.num_pos_feats
        )
        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack(
            [pos_sin, pos_cos], dim=-1
        ).reshape(*xyz.shape[:-1], -1)
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class GPTGroup(nn.Module):
    """FPS + KNN grouping with simplified Morton sorting for PointGPT."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def simplied_morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = [idx_base]
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group
        ).transpose(1, 2).contiguous().view(
            batch_size * self.num_group, self.num_group
        )
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group
        ).transpose(1, 2).contiguous()
        for _ in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last = distances_batch[sorted_indices_list[-1]]
            closest = torch.argmin(distances_to_last, dim=-1) + idx_base
            sorted_indices_list.append(closest)
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group
            ).transpose(1, 2).contiguous().view(
                batch_size * self.num_group, self.num_group
            )
            distances_batch[closest] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group
            ).transpose(1, 2).contiguous()
        return torch.stack(sorted_indices_list, dim=-1).view(-1)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(
            0, batch_size, device=xyz.device
        ).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, 3
        )[sorted_indices, :, :].view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        center = center.view(
            batch_size * self.num_group, 3
        )[sorted_indices, :].view(
            batch_size, self.num_group, 3
        ).contiguous()

        return neighborhood, center

"""
Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework

Paper: ICLR 2022
Authors: Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, Yun Fu
Source: Implementation adapted from: https://github.com/ma-xu/pointMLP-pytorch
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .pointmlp_utils import (
    LocalGrouper, PreExtraction, PosExtraction,
    ConvBNReLU1D, ConvBNReLURes1D,
    index_points, get_activation,
)


def square_distance(src, dst):
    """Pairwise squared distance. src [B,N,C], dst [B,M,C] -> [B,N,M]."""
    return torch.cdist(src.float(), dst.float()).pow(2)
from ..build import MODELS
from ..base_seg_model import BaseSegModel


class PointMLPFeaturePropagation(nn.Module):
    """3-NN interpolation + fuse + residual MLP extraction (matching original)."""

    def __init__(self, in_channel, out_channel, blocks=1, groups=1,
                 res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias, activation=activation)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias,
                                        activation=activation)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, N, 3] dense, xyz2: [B, S, 3] sparse
        points1: [B, D', N] skip, points2: [B, D'', S] from deeper
        Returns: [B, out_C, N]
        """
        points2 = points2.permute(0, 2, 1)  # [B, S, D'']
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)
            interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1)  # [B, C, N]
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


@MODELS.register_module()
class PointMLP_Seg(BaseSegModel):
    """PointMLP segmentation — encoder-decoder with residual MLP blocks."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        points = config.get('points', 2048)
        embed_dim = config.get('embed_dim', 64)
        groups = config.get('groups', 1)
        res_expansion = config.get('res_expansion', 1.0)
        activation = config.get('activation', 'relu')
        bias = config.get('bias', True)
        use_xyz = config.get('use_xyz', True)
        normalize = config.get('normalize', 'anchor')
        dim_expansion = config.get('dim_expansion', [2, 2, 2, 2])
        pre_blocks = config.get('pre_blocks', [2, 2, 2, 2])
        pos_blocks = config.get('pos_blocks', [2, 2, 2, 2])
        k_neighbors = config.get('k_neighbors', [32, 32, 32, 32])
        reducers = config.get('reducers', [4, 4, 4, 4])
        de_dims = config.get('de_dims', [512, 256, 128, 128])
        de_blocks = config.get('de_blocks', [4, 4, 4, 4])
        gmp_dim = config.get('gmp_dim', 64)
        cls_dim = config.get('cls_dim', 64)
        dropout = config.get('dropout', 0.5)

        stages = len(pre_blocks)
        self.stages = stages

        # Input embedding: xyz(3) + normals(3) = 6 → embed_dim
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)

        # Encoder
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = points
        en_dims = [last_channel]

        for i in range(stages):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = anchor_points // reducers[i]
            self.local_grouper_list.append(
                LocalGrouper(last_channel, anchor_points, k_neighbors[i], use_xyz, normalize))
            self.pre_blocks_list.append(
                PreExtraction(last_channel, out_channel, pre_blocks[i], groups=groups,
                              res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz))
            self.pos_blocks_list.append(
                PosExtraction(out_channel, pos_blocks[i], groups=groups,
                              res_expansion=res_expansion, bias=bias, activation=activation))
            last_channel = out_channel
            en_dims.append(last_channel)

        # Decoder
        self.decode_list = nn.ModuleList()
        en_dims_rev = list(reversed(en_dims))
        de_dims_full = [en_dims_rev[0]] + list(de_dims)
        for i in range(len(en_dims_rev) - 1):
            self.decode_list.append(
                PointMLPFeaturePropagation(
                    de_dims_full[i] + en_dims_rev[i + 1], de_dims_full[i + 1],
                    blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                    bias=bias, activation=activation))

        # Global max pooling from each encoder scale
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims_rev:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims_rev), gmp_dim, bias=bias, activation=activation)

        # Class label projection
        if self.use_cls_label:
            self.cls_map = nn.Sequential(
                ConvBNReLU1D(self.num_obj_classes, cls_dim, bias=bias, activation=activation),
                ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation),
            )
            label_dim = cls_dim
        else:
            self.cls_map = None
            label_dim = 0

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim + label_dim + de_dims_full[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.seg_classes, 1, bias=bias),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()
        # pts: [B, N, C]

        B, N, C = pts.shape
        xyz = pts[:, :, :3].contiguous()

        # Input: concat xyz + xyz as "normals" (or real normals if C=6)
        if C >= 6:
            x = pts[:, :, :6].permute(0, 2, 1).contiguous()
        else:
            x = torch.cat([pts[:, :, :3], pts[:, :, :3]], dim=-1).permute(0, 2, 1).contiguous()

        x = self.embedding(x)  # [B, embed_dim, N]

        xyz_list = [xyz]
        x_list = [x]

        # Encoder
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)

        # Decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)

        # Global context from all scales
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))  # [B, gmp_dim, 1]

        # Concat decoder output + global + cls_label
        parts = [x, global_context.expand(-1, -1, x.shape[-1])]
        if self.use_cls_label and cls_label is not None and self.cls_map is not None:
            cls_token = self.cls_map(cls_label.unsqueeze(-1))  # [B, cls_dim, 1]
            parts.append(cls_token.expand(-1, -1, x.shape[-1]))

        x = torch.cat(parts, dim=1)
        x = self.classifier(x)  # [B, seg_classes, N]
        return F.log_softmax(x, dim=1)

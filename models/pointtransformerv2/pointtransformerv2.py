"""
Point Transformer V2: Grouped Vector Attention and Partition-based Pooling

Paper: NeurIPS 2022
Authors: Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao
Source: Implementation adapted from: https://github.com/Pointcept/PointTransformerV2
License: MIT
"""

import torch
import torch.nn as nn

from .utils import (
    PointBatchNorm,
    GVAPatchEmbed,
    Encoder,
    offset2batch,
)
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info


class PointTransformerV2ClsBase(nn.Module):
    """
    Point Transformer V2 backbone for classification.
    Encoder-only architecture with global pooling.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=40,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.15, 0.375, 0.9375),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
    ):
        super(PointTransformerV2ClsBase, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)

        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(grid_sizes)

        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        enc_channels_list = [patch_embed_channels] + list(enc_channels)

        self.enc_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels_list[i],
                embed_channels=enc_channels_list[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]): sum(enc_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            self.enc_stages.append(enc)

        self.final_channels = enc_channels[-1]

        self.cls_head = nn.Sequential(
            nn.Linear(self.final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, data_dict):
        """Extract features from point cloud."""
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        points = [coord, feat, offset]
        points = self.patch_embed(points)

        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)

        coord, feat, offset = points
        return coord, feat, offset

    def forward(self, data_dict):
        """Forward pass with global pooling and classification."""
        coord, feat, offset = self.forward_features(data_dict)

        batch = offset2batch(offset)

        pooled_feats = []
        for i in range(offset.shape[0]):
            if i == 0:
                s_i, e_i = 0, offset[0]
            else:
                s_i, e_i = offset[i - 1], offset[i]
            feat_b = feat[s_i:e_i, :].mean(dim=0, keepdim=True)
            pooled_feats.append(feat_b)

        pooled_feat = torch.cat(pooled_feats, dim=0)

        logits = self.cls_head(pooled_feat)
        return logits


@MODELS.register_module()
class PointTransformerV2(BasePointCloudModel):
    """
    Point Transformer V2 for classification.

    Wraps PointTransformerV2ClsBase with BasePointCloudModel interface.
    Accepts input tensors of shape [B, N, C] or [B, C, N].
    """

    def __init__(self, config, num_classes=40, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointTransformerV2, self).__init__(config, num_classes)

        channels = config.channels_input if hasattr(config, 'channels_input') else channels

        grid_sizes = config.grid_sizes if hasattr(config, 'grid_sizes') else (0.06, 0.15, 0.375, 0.9375)
        enc_depths = config.enc_depths if hasattr(config, 'enc_depths') else (2, 2, 6, 2)
        enc_channels = config.enc_channels if hasattr(config, 'enc_channels') else (96, 192, 384, 512)
        enc_groups = config.enc_groups if hasattr(config, 'enc_groups') else (12, 24, 48, 64)
        drop_path_rate = config.drop_path_rate if hasattr(config, 'drop_path_rate') else 0.3

        self.backbone = PointTransformerV2ClsBase(
            in_channels=channels,
            num_classes=num_classes,
            grid_sizes=grid_sizes,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_groups=enc_groups,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, xyz):
        """
        Forward pass.

        Args:
            xyz: Input point cloud tensor of shape [B, N, C] or [B, C, N]

        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        B, N, C = xyz.shape

        coord = xyz.reshape(-1, C)

        offset = torch.arange(1, B + 1, device=xyz.device) * N
        offset = offset.int()

        input_dict = {
            "coord": coord[:, :3],
            "feat": coord,
            "offset": offset
        }

        return self.backbone(input_dict)


@MODELS.register_module()
class PointTransformerV2Cls(PointTransformerV2):
    """Alias for PointTransformerV2 classification model."""
    pass

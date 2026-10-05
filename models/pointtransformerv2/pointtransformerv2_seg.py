"""
Point Transformer V2: Grouped Vector Attention and Partition-based Pooling

Paper: NeurIPS 2022
Authors: Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao
Source: Implementation adapted from: https://github.com/Pointcept/PointTransformerV2
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    PointBatchNorm,
    GVAPatchEmbed,
    Encoder,
    Decoder,
    offset2batch,
    batch2offset,
)
from ..build import MODELS
from ..base_seg_model import BaseSegModel


class PointTransformerV2SegBase(nn.Module):
    """Point Transformer V2 encoder-decoder backbone."""

    def __init__(
        self,
        in_channels=3,
        num_classes=50,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)

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
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]

        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes),
        )

    def forward(self, coord, feat, offset):
        points = [coord, feat, offset.int()]
        points = self.patch_embed(points)

        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)
            skips.append([points])

        points = skips.pop(-1)[0]
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)

        coord, feat, offset = points
        return self.seg_head(feat)  # [N_total, num_classes]


@MODELS.register_module()
class PointTransformerV2_Seg(BaseSegModel):
    """Point Transformer V2 segmentation model for LIDARLearn."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)

        self.backbone = PointTransformerV2SegBase(
            in_channels=channels,
            num_classes=self.seg_classes,
            grid_sizes=config.get('grid_sizes', (0.06, 0.12, 0.24, 0.48)),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()

        B, N, C = pts.shape

        # Convert batched -> packed (coord always 3D, feat may have more channels)
        coord = pts[:, :, :3].reshape(-1, 3).contiguous()
        feat = pts.reshape(-1, C).contiguous()
        offset = torch.arange(1, B + 1, device=pts.device).int() * N

        out = self.backbone(coord, feat, offset)  # [B*N, seg_classes]

        # Convert packed -> batched [B, seg_classes, N]
        out = out.reshape(B, N, -1).permute(0, 2, 1).contiguous()
        return F.log_softmax(out, dim=1)

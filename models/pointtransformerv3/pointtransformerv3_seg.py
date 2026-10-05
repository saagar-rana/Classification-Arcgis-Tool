"""
Point Transformer V3: Simpler, Faster, Stronger

Paper: CVPR 2024
Authors: Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao
Source: Implementation adapted from: https://github.com/Pointcept/PointTransformerV3
License: MIT
"""

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .utils import (
    Point,
    PointModule,
    PointSequential,
    Block,
    SerializedPooling,
    SerializedUnpooling,
    Embedding,
    offset2batch,
    batch2offset,
)
from ..build import MODELS
from ..base_seg_model import BaseSegModel


class PointTransformerV3SegBase(PointModule):
    """Point Transformer V3 encoder-decoder backbone."""

    def __init__(
        self,
        in_channels=3,
        num_classes=50,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # Encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]):sum(enc_depths[:s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # Decoder
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        dec_channels_full = list(dec_channels) + [enc_channels[-1]]
        self.dec = PointSequential()
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                sum(dec_depths[:s]):sum(dec_depths[:s + 1])
            ]
            dec_drop_path_.reverse()
            dec = PointSequential()
            dec.add(
                SerializedUnpooling(
                    in_channels=dec_channels_full[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels_full[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels_full[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels_full[0], dec_channels_full[0]),
            nn.BatchNorm1d(dec_channels_full[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels_full[0], num_classes),
        )

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)

        return self.seg_head(point.feat)


@MODELS.register_module()
class PointTransformerV3_Seg(BaseSegModel):
    """Point Transformer V3 segmentation model for LIDARLearn."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)

        self.backbone = PointTransformerV3SegBase(
            in_channels=channels,
            num_classes=self.seg_classes,
            enable_flash=config.get('enable_flash', False),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()

        B, N, C = pts.shape

        # Convert batched -> packed Point format (coord always 3D)
        coord = pts[:, :, :3].reshape(-1, 3).contiguous()
        feat = pts.reshape(-1, C).contiguous()
        offset = torch.arange(1, B + 1, device=pts.device).int() * N
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)

        # Grid coord for sparse conv (shift to non-negative, then quantize)
        grid_size = 0.01
        coord_shifted = coord - coord.min(0)[0]
        grid_coord = torch.div(coord_shifted, grid_size, rounding_mode='trunc').int()

        data_dict = dict(
            coord=coord,
            feat=feat,
            batch=batch,
            grid_coord=grid_coord,
            offset=offset,
        )

        out = self.backbone(data_dict)  # [B*N, seg_classes]

        # Convert packed -> batched [B, seg_classes, N]
        out = out.reshape(B, N, -1).permute(0, 2, 1).contiguous()
        return F.log_softmax(out, dim=1)

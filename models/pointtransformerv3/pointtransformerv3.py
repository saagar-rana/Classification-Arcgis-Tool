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
import torch_scatter

from .utils import (
    Point,
    PointModule,
    PointSequential,
    Block,
    SerializedPooling,
    Embedding,
    offset2batch,
    batch2offset,
)
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


class PointTransformerV3Base(PointModule):
    """
    Point Transformer V3 backbone for classification.
    Encoder-only architecture with global pooling.
    """

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,  # Disabled by default for compatibility
        upcast_attention=True,
        upcast_softmax=True,
        num_classes=40,
        grid_size=0.01,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders
        self.grid_size = grid_size
        self.num_classes = num_classes

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # Norm layers
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
                sum(enc_depths[:s]): sum(enc_depths[: s + 1])
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

        self.final_channels = enc_channels[-1]

        # Classification head
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

    def forward(self, data_dict):
        """Forward pass with global pooling and classification."""
        # Create Point structure
        point = Point(data_dict)
        point["grid_size"] = self.grid_size

        # Serialization and sparsification
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # Embedding and encoding
        point = self.embedding(point)
        point = self.enc(point)

        # Global pooling per batch
        feat = point.feat
        offset = point.offset

        pooled_feats = []
        for i in range(offset.shape[0]):
            if i == 0:
                s_i, e_i = 0, offset[0]
            else:
                s_i, e_i = offset[i - 1], offset[i]
            feat_b = feat[s_i:e_i, :].mean(dim=0, keepdim=True)
            pooled_feats.append(feat_b)

        pooled_feat = torch.cat(pooled_feats, dim=0)

        # Classification
        logits = self.cls_head(pooled_feat)
        return logits


@MODELS.register_module()
class PointTransformerV3(BasePointCloudModel):
    """
    Point Transformer V3 for classification.

    Wraps PointTransformerV3Base with BasePointCloudModel interface.
    Accepts input tensors of shape [B, N, C] or [B, C, N].
    """

    def __init__(self, config, num_classes=40, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointTransformerV3, self).__init__(config, num_classes)

        channels = config.channels_input if hasattr(config, 'channels_input') else channels

        # Model configuration
        grid_size = config.grid_size if hasattr(config, 'grid_size') else 0.01
        enc_depths = config.enc_depths if hasattr(config, 'enc_depths') else (2, 2, 2, 6, 2)
        enc_channels = config.enc_channels if hasattr(config, 'enc_channels') else (32, 64, 128, 256, 512)
        enc_num_head = config.enc_num_head if hasattr(config, 'enc_num_head') else (2, 4, 8, 16, 32)
        enc_patch_size = config.enc_patch_size if hasattr(config, 'enc_patch_size') else (48, 48, 48, 48, 48)
        stride = config.stride if hasattr(config, 'stride') else (2, 2, 2, 2)
        drop_path = config.drop_path_rate if hasattr(config, 'drop_path_rate') else 0.3
        enable_flash = config.enable_flash if hasattr(config, 'enable_flash') else False

        self.backbone = PointTransformerV3Base(
            in_channels=channels,
            num_classes=num_classes,
            grid_size=grid_size,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            stride=stride,
            drop_path=drop_path,
            enable_flash=enable_flash,
        )

    def forward(self, xyz):
        """
        Forward pass.

        Args:
            xyz: Input point cloud tensor of shape [B, N, C] or [B, C, N]

        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        # Handle input format
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        B, N, C = xyz.shape

        # Flatten batch dimension
        coord = xyz.reshape(-1, C)

        # Create offset for batch handling
        offset = torch.arange(1, B + 1, device=xyz.device) * N
        offset = offset.long()

        # Prepare input dict
        input_dict = {
            "coord": coord[:, :3],
            "feat": coord,
            "offset": offset,
        }

        return self.backbone(input_dict)


@MODELS.register_module()
class PointTransformerV3Cls(PointTransformerV3):
    """Alias for PointTransformerV3 classification model."""
    pass

"""
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

Paper: NeurIPS 2017
Authors: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas
Source: Implementation adapted from: https://github.com/erikwijmans/Pointnet2_PyTorch
License: Public Domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import (
    PointnetSAModuleMSG,
    PointnetSAModule,
    PointnetFPModule,
)

from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PointNet2_MSG_Seg(BaseSegModel):
    """PointNet++ MSG segmentation model with SA encoder + FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        dropout = config.get('dropout', 0.5)

        # ── Encoder (Set Abstraction) ──
        self.SA_modules = nn.ModuleList()

        # First SA: use_xyz=True adds 3 xyz channels internally.
        # When input has no extra features (xyz-only), c_in=0 so MLP input = 0+3 = 3.
        # When input has extra features (e.g. normals), c_in = extra_channels.
        c_in = 0  # use_xyz=True handles xyz channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=True,
            )
        )
        c_out_0 = 32 + 64  # 96

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128], [c_out_0, 64, 96, 128]],
                use_xyz=True,
            )
        )
        c_out_1 = 128 + 128  # 256

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_1, 128, 196, 256], [c_out_1, 128, 196, 256]],
                use_xyz=True,
            )
        )
        c_out_2 = 256 + 256  # 512

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_out_2, 256, 256, 512], [c_out_2, 256, 384, 512]],
                use_xyz=True,
            )
        )
        c_out_3 = 512 + 512  # 1024

        # ── Decoder (Feature Propagation) ──
        # Loop runs i = -1, -2, -3, -4 → FP_modules[-1] runs first (deepest).
        # So FP_modules[3] = deepest, FP_modules[0] = shallowest.
        #
        # FP[3] (i=-1): interp l_feat[4](1024ch) + l_feat[3](512ch) = 1536 -> 512
        # FP[2] (i=-2): interp FP_out(512ch) + l_feat[2](256ch) = 768 -> 256
        # FP[1] (i=-3): interp FP_out(256ch) + l_feat[1](96ch) = 352 -> 128
        # FP[0] (i=-4): interp FP_out(128ch) + l_feat[0](None) = 128 -> 128
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128, 128, 128]))                  # shallowest
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 128]))        # 352 -> 128
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256]))        # 768 -> 256
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))   # deepest: 1536 -> 512

        # ── Segmentation head (same structure as PointCentricSegWrapper) ──
        cls_label_dim = 64 if self.use_cls_label else 0
        self.cls_proj = nn.Sequential(
            nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        ) if self.use_cls_label else None

        in_dim = 128 + cls_label_dim
        self.head = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] -> [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()

        B, N, C = pts.shape
        xyz = pts[..., :3].contiguous()
        features = pts[..., 3:].transpose(1, 2).contiguous() if C > 3 else None

        # Encoder
        l_xyz = [xyz]
        l_features = [features]
        for sa in self.SA_modules:
            li_xyz, li_features = sa(l_xyz[-1], l_features[-1])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Decoder (reverse order)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Per-point features: [B, 128, N]
        per_point = l_features[0]

        # Append class label if part segmentation
        if self.use_cls_label and cls_label is not None and self.cls_proj is not None:
            # cls_label: [B, num_obj_classes] -> [B, num_obj_classes, N]
            cls_expand = cls_label.unsqueeze(-1).expand(-1, -1, N)
            cls_feat = self.cls_proj(cls_expand)  # [B, 64, N]
            per_point = torch.cat([per_point, cls_feat], dim=1)

        logits = self.head(per_point)  # [B, seg_classes, N]
        return F.log_softmax(logits, dim=1)

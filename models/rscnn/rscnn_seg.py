"""
Relation-Shape Convolutional Neural Network for Point Cloud Analysis

Paper: CVPR 2019
Authors: Yongcheng Liu, Bin Fan, Shiming Xiang, Chunhong Pan
Source: Implementation adapted from: https://github.com/Yochengliu/Relation-Shape-CNN
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule, PointnetFPModule
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class RSCNN_Seg(BaseSegModel):
    """RS-CNN MSN segmentation — relation-shape SA encoder + FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        input_channels = config.get('input_channels', 0)
        relation_prior = config.get('relation_prior', 1)
        use_xyz = config.get('use_xyz', True)
        dropout = config.get('dropout', 0.5)

        # Encoder (4 SA-MSG stages with relation-shape conv)
        self.SA_modules = nn.ModuleList()

        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024, radii=[0.075, 0.1, 0.125], nsamples=[16, 32, 48],
                mlps=[[c_in, 64], [c_in, 64], [c_in, 64]],
                first_layer=True, use_xyz=use_xyz, relation_prior=relation_prior))
        c_out_0 = 64 * 3  # 192

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256, radii=[0.1, 0.15, 0.2], nsamples=[16, 48, 64],
                mlps=[[c_out_0, 128], [c_out_0, 128], [c_out_0, 128]],
                use_xyz=use_xyz, relation_prior=relation_prior))
        c_out_1 = 128 * 3  # 384

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64, radii=[0.2, 0.3, 0.4], nsamples=[16, 32, 48],
                mlps=[[c_out_1, 256], [c_out_1, 256], [c_out_1, 256]],
                use_xyz=use_xyz, relation_prior=relation_prior))
        c_out_2 = 256 * 3  # 768

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16, radii=[0.4, 0.6, 0.8], nsamples=[16, 24, 32],
                mlps=[[c_out_2, 512], [c_out_2, 512], [c_out_2, 512]],
                use_xyz=use_xyz, relation_prior=relation_prior))
        c_out_3 = 512 * 3  # 1536

        # Global pooling modules
        self.global_sa1 = PointnetSAModule(nsample=16, mlp=[c_out_3, 128], use_xyz=use_xyz)
        self.global_sa2 = PointnetSAModule(nsample=64, mlp=[c_out_2, 128], use_xyz=use_xyz)

        # Decoder (4 FP stages)
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))      # shallowest
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))          # deepest

        # Seg head: local(128) + global1(128) + global2(128) + cls_label(16) = 400
        label_dim = self.num_obj_classes if self.use_cls_label else 0
        head_in = 128 + 128 + 128 + label_dim
        self.FC_layer = nn.Sequential(
            nn.Conv1d(head_in, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()

        xyz, features = self._break_up_pc(pts)

        # Encoder
        l_xyz, l_features = [xyz], [features]
        for i in range(4):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            if li_xyz is not None:
                idx = np.arange(li_xyz.size(1))
                np.random.shuffle(idx)
                li_xyz = li_xyz[:, idx, :]
                li_features = li_features[:, :, idx]
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Global pooling
        _, global_feat1 = self.global_sa1(l_xyz[4], l_features[4])  # from deepest
        _, global_feat2 = self.global_sa2(l_xyz[3], l_features[3])  # from level 3

        # Decoder — propagate features from deepest level back to level 0.
        # FP_modules[0] = shallowest (level1->level0), FP_modules[3] = deepest (level4->level3)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        B, _, N = l_features[0].shape

        # Concat: local + global1 + global2 + cls_label
        parts = [
            l_features[0],
            global_feat1.expand(-1, -1, N),
            global_feat2.expand(-1, -1, N),
        ]
        if self.use_cls_label and cls_label is not None:
            cls = cls_label.unsqueeze(-1).expand(-1, -1, N)
            parts.append(cls)

        out = torch.cat(parts, dim=1)
        logits = self.FC_layer(out)
        return F.log_softmax(logits, dim=1)

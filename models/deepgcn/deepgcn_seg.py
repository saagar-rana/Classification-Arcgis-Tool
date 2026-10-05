"""
DeepGCNs: Can GCNs Go as Deep as CNNs?

Paper: ICCV 2019
Authors: Guohao Li, Matthias Muller, Ali Thabet, Bernard Ghanem
Source: Implementation adapted from: https://github.com/lightaime/deep_gcns_torch
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_nn import BasicConv, Seq
from .torch_edge import DilatedKnnGraph, DenseDilatedKnnGraph
from .torch_vertex import GraphConv2d, ResDynBlock2d, DenseDynBlock2d, PlainDynBlock2d
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class DeepGCN_Seg(BaseSegModel):
    """DeepGCN segmentation — preserves all N points, no encoder-decoder needed."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        gcn_channels = config.get('gcn_channels', 64)
        k = config.get('k', 9)
        act = config.get('act', 'relu')
        norm = config.get('norm', 'batch')
        bias = config.get('bias', True)
        knn = config.get('knn', 'matrix')
        epsilon = config.get('epsilon', 0.2)
        stochastic = config.get('stochastic', False)
        conv = config.get('conv', 'edge')
        c_growth = config.get('c_growth', 64)
        emb_dims = config.get('emb_dims', 1024)
        dropout = config.get('dropout', 0.5)
        n_blocks = config.get('n_blocks', 14)
        block_type = config.get('block_type', 'res')
        use_dilation = config.get('use_dilation', True)

        self.n_blocks = n_blocks

        if knn == 'matrix':
            self.knn_graph = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        else:
            self.knn_graph = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(channels, gcn_channels, conv, act, norm, bias=False)

        if block_type.lower() == 'res':
            if use_dilation:
                self.backbone = Seq(*[
                    ResDynBlock2d(gcn_channels, k, i + 1, conv, act, norm,
                                 bias, stochastic, epsilon, knn)
                    for i in range(n_blocks - 1)
                ])
            else:
                self.backbone = Seq(*[
                    ResDynBlock2d(gcn_channels, k, 1, conv, act, norm,
                                 bias, stochastic, epsilon, knn)
                    for _ in range(n_blocks - 1)
                ])
            fusion_dims = int(gcn_channels + c_growth * (n_blocks - 1))
        elif block_type.lower() == 'dense':
            self.backbone = Seq(*[
                DenseDynBlock2d(gcn_channels + c_growth * i, c_growth, k,
                                1 + i, conv, act, norm, bias, stochastic, epsilon, knn)
                for i in range(n_blocks - 1)
            ])
            fusion_dims = int(
                (gcn_channels + gcn_channels + c_growth * (n_blocks - 1)) * n_blocks // 2
            )
        else:
            stochastic = False
            self.backbone = Seq(*[
                PlainDynBlock2d(gcn_channels, k, 1, conv, act, norm,
                                bias, stochastic, epsilon, knn)
                for _ in range(n_blocks - 1)
            ])
            fusion_dims = int(gcn_channels + c_growth * (n_blocks - 1))

        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm, bias=False)

        # Seg prediction: global(max+avg=2*emb_dims) + per_point(emb_dims) = 3*emb_dims
        self.prediction = Seq(*[
            BasicConv([emb_dims * 3, 512], 'leakyrelu', norm, drop=dropout),
            BasicConv([512, 256], 'leakyrelu', norm, drop=dropout),
            BasicConv([256, self.seg_classes], None, None),
        ])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1).contiguous()

        # Encoder: all N points preserved
        feats = [self.head(pts, self.knn_graph(pts[:, 0:3]))]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)

        # Fusion: concat all block outputs → emb_dims
        fusion = self.fusion_block(feats)  # [B, emb_dims, N, 1]

        # Global pool broadcast + per-point features
        x1 = F.adaptive_max_pool2d(fusion, 1)   # [B, emb_dims, 1, 1]
        x2 = F.adaptive_avg_pool2d(fusion, 1)   # [B, emb_dims, 1, 1]
        global_pool = torch.cat((x1, x2), dim=1)  # [B, 2*emb_dims, 1, 1]
        global_pool = global_pool.expand(-1, -1, fusion.shape[2], -1)  # broadcast to N

        # Concat global + local
        cat_feat = torch.cat((global_pool, fusion), dim=1)  # [B, 3*emb_dims, N, 1]

        # Prediction
        out = self.prediction(cat_feat).squeeze(-1)  # [B, seg_classes, N]
        return F.log_softmax(out, dim=1)

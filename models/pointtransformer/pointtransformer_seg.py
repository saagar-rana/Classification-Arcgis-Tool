"""
Point Transformer

Paper: ICCV 2021
Authors: Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip H. S. Torr, Vladlen Koltun
Source: Implementation adapted from: https://github.com/POSTECH-CVLab/point-transformer
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import TransitionDown, TransitionUp, Bottleneck
from ..build import MODELS
from ..base_seg_model import BaseSegModel


class PointTransformerSegBase(nn.Module):
    """Point Transformer encoder-decoder for dense per-point prediction."""

    def __init__(self, block, blocks, in_channels=3, num_classes=50,
                 num_shape_classes=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_shape_classes = num_shape_classes
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [8, 16, 16, 16, 16]

        # Encoder
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes,
                                   stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes,
                                   stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes,
                                   stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes,
                                   stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes,
                                   stride=stride[4], nsample=nsample[4])

        # Decoder
        self.dec5 = self._make_dec(block, planes[4], 1, share_planes,
                                   nsample=nsample[4], is_head=True,
                                   num_shape_classes=num_shape_classes)
        self.dec4 = self._make_dec(block, planes[3], 1, share_planes,
                                   nsample=nsample[3])
        self.dec3 = self._make_dec(block, planes[2], 1, share_planes,
                                   nsample=nsample[2])
        self.dec2 = self._make_dec(block, planes[1], 1, share_planes,
                                   nsample=nsample[1])
        self.dec1 = self._make_dec(block, planes[0], 1, share_planes,
                                   nsample=nsample[0])

        # Seg head
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16,
                  is_head=False, num_shape_classes=None):
        layers = [
            TransitionUp(
                self.in_planes,
                None if is_head else planes * block.expansion,
                num_shape_classes if is_head else None,
            )
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, coord, feat, offset, cls_token=None):
        p0, x0, o0 = coord, feat, offset.int()

        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        # Decoder
        if self.num_shape_classes is not None and cls_token is not None:
            x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5], y=cls_token), o5])[1]
        else:
            x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]

        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return self.cls(x1)  # [N_total, num_classes]


@MODELS.register_module()
class PointTransformer_Seg(BaseSegModel):
    """Point Transformer V1 segmentation model for LIDARLearn."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        num_shape_classes = self.num_obj_classes if self.use_cls_label else None

        self.backbone = PointTransformerSegBase(
            Bottleneck, [1, 2, 3, 5, 2],
            in_channels=channels,
            num_classes=self.seg_classes,
            num_shape_classes=num_shape_classes,
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()

        B, N, C = pts.shape

        # Convert batched [B, N, C] -> packed [B*N, 3] + offset
        coord = pts[:, :, :3].reshape(-1, 3).contiguous()
        feat = pts.reshape(-1, C).contiguous()
        offset = torch.arange(1, B + 1, device=pts.device).int() * N

        # Class token for part segmentation
        cls_token = None
        if self.use_cls_label and cls_label is not None:
            # cls_label: [B, num_obj_classes] one-hot -> [B] class index
            cls_token = cls_label.argmax(dim=1)  # [B]

        # Run encoder-decoder
        out = self.backbone(coord, feat, offset, cls_token)  # [B*N, seg_classes]

        # Convert packed [B*N, seg_classes] -> batched [B, seg_classes, N]
        out = out.reshape(B, N, -1).permute(0, 2, 1).contiguous()

        return F.log_softmax(out, dim=1)

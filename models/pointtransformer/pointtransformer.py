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

from .utils import TransitionDown, Bottleneck
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

class PointTransformerClsBase(nn.Module):
    def __init__(self, block, blocks, in_channels=3, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

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

        self.cls = nn.Sequential(nn.Linear(planes[4], 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(256, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(128, num_classes))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward_features(self, input_dict):
        p0 = input_dict["coord"]
        x0 = input_dict["feat"]
        o0 = input_dict["offset"].int()
        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        x = []
        for i in range(o5.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o5[0], o5[0]
            else:
                s_i, e_i, cnt = o5[i - 1], o5[i], o5[i] - o5[i - 1]
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x.append(x_b)
        x = torch.cat(x, 0)
        return x

    def forward(self, input_dict):
        x = self.forward_features(input_dict)
        x = self.cls(x)
        return x

@MODELS.register_module()
class PointTransformer(BasePointCloudModel):
    def __init__(self, config, num_classes=40, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointTransformer, self).__init__(config, num_classes)

        channels = config.channels_input

        self.backbone = PointTransformerClsBase(Bottleneck, [1, 2, 3, 5, 2], in_channels=channels, num_classes=num_classes)

    def forward(self, xyz):
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        B, N, C = xyz.shape

        coord = xyz.reshape(-1, C)

        offset = torch.arange(1, B + 1, device=xyz.device) * N
        offset = offset.int()

        input_dict = {
            "coord": coord,
            "feat": None,
            "offset": offset
        }

        return self.backbone(input_dict)

@MODELS.register_module()
class PointTransformer50(PointTransformer):
    pass

"""
PointCNN: Convolution On X-Transformed Points

Paper: NeurIPS 2018
Authors: Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen
Source: Implementation adapted from: https://github.com/yangyanli/PointCNN
License: MIT
"""

import torch
from torch import nn
import torch.nn.functional as F

from .util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from .model import RandPointCNN
from .util_layers import Dense
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

def AbbPointCNN(a, b, c, d, e): return RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)

@MODELS.register_module()
class PointCNN(BasePointCloudModel):
    def __init__(self, config, num_classes=40, channels=3, **kwargs):
        super(PointCNN, self).__init__(config, num_classes)

        channels = config.channels_input

        self.pcnn1 = AbbPointCNN(channels, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, self.num_classes, with_bn=False, activation=None)
        )

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 3:
            x = x.permute(0, 2, 1)

        x = (x, x)

        x = self.pcnn1(x)

        x = self.pcnn2(x)[1]

        logits = self.fcn(x)

        logits_mean = torch.mean(logits, dim=1)

        return logits_mean

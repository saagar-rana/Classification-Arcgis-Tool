"""
PPFNet: Global Context Aware Local Features for Robust 3D Point Matching

Paper: CVPR 2018
Authors: Haowen Deng, Tolga Birdal, Slobodan Ilic
Source: Implementation adapted from: https://github.com/vinits5/learning3d
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ppfnet_util import sample_and_group_multi, get_prepool, get_postpool
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}

@MODELS.register_module()
class PPFNet(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PPFNet, self).__init__(config, num_classes)

        features = config.features
        emb_dims = config.emb_dims
        radius = config.radius
        num_neighbors = config.num_neighbors
        dropout1 = config.dropout1
        dropout2 = config.dropout2

        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])

        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])
        self.prepool = get_prepool(raw_dim, emb_dims * 2)
        self.postpool = get_postpool(emb_dims * 2, emb_dims)

        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout1)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout2)
        self.linear3 = nn.Linear(256, self.num_classes)

    def forward(self, xyz, normals=None):
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        normals = torch.zeros_like(xyz).to(xyz.device)

        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]

        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        new_feat = fused_input_feat.permute(0, 3, 2, 1)
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]

        post_feat = self.postpool(pooled_feat)
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)
        cluster_feat = cluster_feat.permute(0, 2, 1)
        cluster_feat = F.adaptive_max_pool1d(cluster_feat, 1).view(cluster_feat.size(0), -1)
        x = F.leaky_relu(self.bn6(self.linear1(cluster_feat)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

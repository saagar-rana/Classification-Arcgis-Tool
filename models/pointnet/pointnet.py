"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

Paper: CVPR 2017
Authors: Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
Source: Implementation adapted from: https://github.com/charlesq34/pointnet
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TNet, FeatureTransformNet
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PointNet(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointNet, self).__init__(config, num_classes)

        input_channels = config.input_channels
        emb_dims = config.emb_dims
        dropout = config.dropout
        feature_transform = config.feature_transform

        self.input_channels = input_channels
        self.emb_dims = emb_dims
        self.feature_transform = feature_transform

        self.stn = TNet(k=input_channels)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        if feature_transform:
            self.fstn = FeatureTransformNet(k=64)

        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.shape[-1] == 3 or x.shape[-1] == self.input_channels:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        trans_input = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_input)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        x = self.dropout2(x)
        x = self.linear3(x)

        return x

    def get_feature_transform_regularization(self):
        if not self.feature_transform:
            return 0

        identity = torch.eye(64, device=self.fstn.fc3.weight.device)
        trans_feat = self.fstn.get_transform_matrix()

        loss = torch.mean(torch.norm(identity - torch.bmm(trans_feat.transpose(2, 1), trans_feat), dim=(1, 2)))
        return loss * 0.001

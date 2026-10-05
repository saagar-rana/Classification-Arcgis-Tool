"""
SO-Net: Self-Organizing Network for Point Cloud Analysis

Paper: CVPR 2018
Authors: Jiaxin Li, Ben M. Chen, Gim Hee Lee
Source: Implementation adapted from: https://github.com/lijx10/SO-Net
License: MIT
"""

import math
import torch
import torch.nn as nn

import index_max
from .som import BatchSOM
from . import operations
from .layers import *
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

class Transformer(nn.Module):
    def __init__(self, opt=None):
        super(Transformer, self).__init__()
        if opt is None:
            opt = type('obj', (object,), {
                'activation': 'relu',
                'normalization': 'batch',
                'bn_momentum': 0.1,
                'bn_momentum_decay_step': None,
                'bn_momentum_decay': 0.6,
                'dropout': 0.5
            })()
        self.opt = opt

        self.first_pointnet = PointNet(3, (32, 64, 128), activation=self.opt.activation, normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(128 + 128, (256, 256), activation=self.opt.activation, normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.fc1 = MyLinear(256, 128, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(128, 64, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(64, 1, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, x, sn=None, epoch=None):
        first_pn_out = self.first_pointnet(x, epoch)
        feature_1, _ = torch.max(first_pn_out, dim=2, keepdim=False)

        second_pn_out = self.second_pointnet(torch.cat((first_pn_out, feature_1.unsqueeze(2).expand_as(first_pn_out)), dim=1), epoch)
        feature_2, _ = torch.max(second_pn_out, dim=2, keepdim=False)

        fc1_out = self.fc1(feature_2, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)

        sin_theta = torch.tanh(self.fc3(self.fc2_out, epoch))

        return sin_theta

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        self.transformer = Transformer()

        if self.opt.surface_normal == True:
            self.first_pointnet = PointResNet(6, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointResNet(3, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        if self.opt.som_k >= 2:
            self.knnlayer = KNNModule(3 + 384, (512, 512), activation=self.opt.activation, normalization=self.opt.normalization,
                                      momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

            self.final_pointnet = PointNet(3 + 512, (768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.final_pointnet = PointResNet(3 + 384, (512, 512, 768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        rows = int(math.sqrt(self.opt.node_num))
        cols = rows
        self.som_builder = BatchSOM(rows, cols, 3, self.opt.gpu_id, self.opt.batch_size)

        self.zero_pad = torch.nn.ZeroPad2d(padding=1)

    def forward(self, x, sn, node, node_knn_I, is_train=False, epoch=None):
        self.som_builder.node.resize_(node.size()).copy_(node)

        self.mask, mask_row_max, min_idx = self.som_builder.query_topk(x.data, k=self.opt.k)
        mask_row_sum = torch.sum(self.mask, dim=1)
        mask = self.mask.unsqueeze(1)

        x_list, sn_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            sn_list.append(sn)
        x_stack = torch.cat(tuple(x_list), dim=2)
        sn_stack = torch.cat(tuple(sn_list), dim=2)

        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float() + 1e-5)
        self.som_builder.node = cluster_mean
        self.som_node = self.som_builder.node

        node_expanded = self.som_node.data.unsqueeze(2)
        self.centers = torch.sum(mask.float() * node_expanded, dim=3).detach()

        self.x_decentered = (x_stack - self.centers).detach()
        x_augmented = torch.cat((self.x_decentered, sn_stack), dim=1)

        if self.opt.surface_normal == True:
            self.first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            self.first_pn_out = self.first_pointnet(self.x_decentered, epoch)

        M = node.size()[2]
        with torch.cuda.device(self.first_pn_out.get_device()):
            gather_index = index_max.forward_cuda(self.first_pn_out.detach(),
                                                  min_idx.int(),
                                                  M).detach().long()
        self.first_pn_out_masked_max = self.first_pn_out.gather(dim=2, index=gather_index * mask_row_max.unsqueeze(1).long())

        if self.opt.som_k >= 2:
            self.knn_center_1, self.knn_feature_1 = self.knnlayer(self.som_node, self.first_pn_out_masked_max, node_knn_I, self.opt.som_k, self.opt.som_k_type, epoch)

            self.final_pn_out = self.final_pointnet(torch.cat((self.knn_center_1, self.knn_feature_1), dim=1), epoch)
        else:
            self.final_pn_out = self.final_pointnet(torch.cat((self.som_node, self.first_pn_out_masked_max), dim=1), epoch)

        self.feature, _ = torch.max(self.final_pn_out, dim=2, keepdim=False)

        return self.feature

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        self.fc1 = MyLinear(self.feature_num, 512, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(512, 256, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(256, self.opt.classes, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, feature, epoch=None):
        fc1_out = self.fc1(feature, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)
        score = self.fc3(self.fc2_out, epoch)

        return score

@MODELS.register_module()
class SONet(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.num_classes
        super(SONet, self).__init__(config, num_classes)

        self.node_num = config.node_num
        self.feature_num = config.feature_num
        self.k = config.k
        self.som_k = config.som_k
        self.dropout = config.dropout

        self.opt = type('obj', (object,), {
            'classes': self.num_classes,
            'node_num': self.node_num,
            'feature_num': self.feature_num,
            'k': self.k,
            'som_k': self.som_k,
            'som_k_type': 'avg',
            'dropout': self.dropout,
            'activation': 'relu',
            'normalization': 'batch',
            'bn_momentum': 0.1,
            'bn_momentum_decay_step': None,
            'bn_momentum_decay': 0.6,
            'surface_normal': False,
            'batch_size': 16,
            'input_pc_num': 1024,
            'gpu_id': 0,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        })()

        self.encoder = Encoder(self.opt)
        self.classifier = Classifier(self.opt)

    def forward(self, pointcloud):
        if pointcloud.dim() == 3 and pointcloud.shape[1] != 3:
            if pointcloud.shape[2] == 3:
                pointcloud = pointcloud.permute(0, 2, 1)

        B, C, N = pointcloud.shape

        self.opt.batch_size = B
        self.opt.input_pc_num = N

        sn = torch.zeros_like(pointcloud)

        rows = int(math.sqrt(self.node_num))
        cols = rows
        actual_node_num = rows * cols
        input_node = torch.randn(B, 3, actual_node_num).to(pointcloud.device)

        input_node_knn_I = torch.zeros(B, actual_node_num, self.som_k, dtype=torch.long).to(pointcloud.device)

        feature = self.encoder(pointcloud, sn, input_node, input_node_knn_I, is_train=self.training, epoch=None)

        logits = self.classifier(feature, epoch=None)

        return logits

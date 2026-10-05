"""
P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting

Paper: NeurIPS 2022
Authors: Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, Jiwen Lu
Source: Implementation adapted from: https://github.com/wangzy22/P2P
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from easydict import EasyDict

from .encoder import ProjEnc
from .head import MLPHead
from ..build import MODELS
from ..base_model import BasePointCloudModel

@MODELS.register_module()
class P2P(BasePointCloudModel):
    def __init__(self, config, num_classes=40, **kwargs):
        super(P2P, self).__init__(config, num_classes)

        cfg = EasyDict()
        cfg.base_model_variant = config.base_model_variant
        cfg.head_type = config.head_type
        cfg.local_size = config.local_size
        cfg.trans_dim = config.trans_dim
        cfg.graph_dim = config.graph_dim
        cfg.imgblock_dim = config.imgblock_dim
        cfg.img_size = config.img_size
        cfg.obj_size = config.obj_size
        cfg.classes = self.num_classes
        cfg.checkpoint_path = config.checkpoint_path
        cfg.update_type = config.update_type

        cfg.imagenet_default_mean = [0.485, 0.456, 0.406]
        cfg.imagenet_default_std = [0.229, 0.224, 0.225]

        cfg.mlp_mid_channels = config.mlp_mid_channels
        cfg.mlp_dropout_ratio = config.mlp_dropout_ratio

        self.cfg = cfg

        self.enc = ProjEnc(cfg)

        if cfg.checkpoint_path is not None:
            self.base_model = create_model(cfg.base_model_variant, checkpoint_path=cfg.checkpoint_path)
        else:
            self.base_model = create_model(cfg.base_model_variant, pretrained=True)

        if 'resnet' in cfg.base_model_variant:
            self.base_model.num_features = self.base_model.fc.in_features

        if cfg.head_type == 'mlp':
            cls_head = MLPHead(self.base_model.num_features, cfg.classes, cfg.mlp_mid_channels, cfg.mlp_dropout_ratio)
        elif cfg.head_type == 'linear':
            cls_head = nn.Linear(self.base_model.num_features, cfg.classes)
        else:
            raise ValueError('cfg.head_type must be "mlp" or "linear"!')

        if 'convnext' in cfg.base_model_variant:
            self.base_model.head.fc = cls_head
        elif 'resnet' in cfg.base_model_variant:
            self.base_model.fc = cls_head
        else:
            self.base_model.head = cls_head

    def _fix_weight(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        if 'vit' in self.cfg.base_model_variant:
            self.base_model.cls_token.requires_grad = True

        if 'convnext' in self.cfg.base_model_variant:
            for param in self.base_model.head.fc.parameters():
                param.requires_grad = True
        elif 'resnet' in self.cfg.base_model_variant:
            for param in self.base_model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.head.parameters():
                param.requires_grad = True

        if self.cfg.update_type is not None:
            for name, param in self.base_model.named_parameters():
                if self.cfg.update_type in name:
                    param.requires_grad = True

    def forward(self, xyz):
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        original_pc = xyz
        pc = xyz

        img = self.enc(original_pc, pc)

        out = self.base_model(img)

        return out

"""
Registered segmentation model wrappers.

These classes are registered in the MODELS registry so they can be
instantiated from YAML configs through `build_model_from_cfg`. They simply
build an in-tree classification backbone, wrap it with
PointCentricSegWrapper, and expose `forward(pts, cls_label)`.

Example YAML:
    model:
      NAME: PointCentricSeg
      adapter_name: DGCNN
      seg_classes: 50
      num_obj_classes: 16
      use_cls_label: True
      backbone:
        NAME: DGCNN
        channels_input: 3
        k: 20
        k1: 20
        emb_dims: 1024
        dropout: 0.5
        num_classes: 16   # dummy, its cls head is unused
"""

import torch

from ..build import MODELS, build_model_from_cfg
from .point_centric_seg import PointCentricSegWrapper


@MODELS.register_module()
class PointCentricSeg(PointCentricSegWrapper):
    """Registry-friendly wrapper. Config drives everything."""

    def __init__(self, config, **kwargs):
        # Build backbone from nested config
        backbone_cfg = config.backbone
        backbone = build_model_from_cfg(backbone_cfg)

        super().__init__(
            backbone=backbone,
            seg_classes=config.seg_classes,
            adapter_name=config.adapter_name,
            use_cls_label=config.use_cls_label,
            num_obj_classes=config.num_obj_classes,
            dropout=config.dropout,
            config=config,
        )

    def load_model_from_ckpt(self, ckpt_path: str):
        """Alias for backbone checkpoint loading (used by runner_seg)."""
        return self.load_backbone_ckpt(ckpt_path, strict=False)

"""
Base Model Class for Point Cloud Segmentation (per-point classification).

All segmentation models should inherit from BaseSegModel. The forward pass
must return per-point logits. Unlike BasePointCloudModel (which returns
[B, num_classes]), segmentation models return [B, seg_classes, N].

Author: Said Ohamouddou
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSegModel(nn.Module, ABC):
    """
    Abstract base class for point cloud segmentation models.

    Required Methods:
        - forward(pts, cls_label=None): Returns [B, seg_classes, N] log-probabilities
                                        (already log-softmax'd along dim=1).

    Notes on output convention:
        forward() returns log-probabilities (post log_softmax) with shape
        [B, seg_classes, N]. This matches the convention used by
        Point_M2AE_SEG and the PPT segmentation reference. NLL loss
        expects log-probabilities, so we use F.nll_loss in get_loss_acc.
    """

    def __init__(self, config, seg_classes=50, num_obj_classes=16, **kwargs):
        super().__init__()
        self.seg_classes = config.seg_classes
        self.num_obj_classes = config.num_obj_classes
        self.use_cls_label = config.use_cls_label
        self.config = config

    @abstractmethod
    def forward(self, pts: torch.Tensor, cls_label: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pts: [B, C, N] or [B, N, C] (C=3 for xyz; implementations may accept more)
            cls_label: [B, num_obj_classes] one-hot object-category label for part seg.
                       None for semantic segmentation.
        Returns:
            log-probs of shape [B, seg_classes, N]
        """
        pass

    def get_loss_acc(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple:
        """
        Args:
            pred: [B, seg_classes, N] log-probabilities (post log_softmax)
            gt:   [B, N] long tensor of per-point labels
        Returns:
            (loss, acc) where acc is per-point accuracy as a percentage.
        """
        B, C, N = pred.shape
        # NLL loss expects [B, C, N] with class dim at position 1
        loss = F.nll_loss(pred, gt.long())
        pred_choice = pred.argmax(dim=1)  # [B, N]
        acc = (pred_choice == gt).float().mean() * 100.0
        return loss, acc

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

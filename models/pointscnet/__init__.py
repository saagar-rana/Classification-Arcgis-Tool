"""
PointSCNet: Point Cloud Structure and Correlation Learning Based on
            Space-Filling Curve-Guided Sampling

Paper: Chen et al., Symmetry 2022 — https://www.mdpi.com/2073-8994/14/8/1485
Source: https://github.com/Chenguoz/PointSCNet

Clean implementation.
"""

from .network import PointSCNet
from .curve_sampling import HilbertSampler
from .feature_layers import (
    LocalRegionEncoder,
    MultiScaleLocalEncoder,
    FeatureUpsampler
)
from .attention import DualPathAttention

__all__ = [
    'PointSCNet',
    'HilbertSampler',
    'LocalRegionEncoder',
    'MultiScaleLocalEncoder',
    'FeatureUpsampler',
    'DualPathAttention'
]

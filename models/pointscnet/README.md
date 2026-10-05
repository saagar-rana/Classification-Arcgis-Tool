# SCPCloud: Spatial Curve Point Cloud Network

A clean room implementation for 3D point cloud classification using space-filling curves and hierarchical feature learning.

## License

MIT License - See [LICENSE](LICENSE) file.

## Overview

This is an **independent implementation** designed from functional requirements, not derived from any existing codebase. It provides:

- **Space-filling curve sampling**: Uses Hilbert curves for locality-preserving point sampling
- **Hierarchical feature extraction**: Multi-scale local neighborhood encoding
- **Attention mechanisms**: Dual-path (channel + spatial) attention refinement
- **Cross-point interaction**: Global context through curve-sampled reference points

## Installation

```bash
pip install torch numpy
```

## Usage

```python
from clean_room import SCPCloudClassifier

# Create classifier for 40 classes (e.g., ModelNet40)
model = SCPCloudClassifier(
    num_classes=40,
    use_normals=False,
    drop_rate_1=0.4,
    drop_rate_2=0.4
)

# Input: batch of point clouds [B, N, 3] or [B, 3, N]
import torch
points = torch.randn(8, 1024, 3)  # 8 point clouds, 1024 points each

# Forward pass
logits = model(points)  # [8, 40]
predictions = logits.argmax(dim=-1)
```

## Architecture

```
Input Point Cloud [B, N, 3]
         │
         ▼
┌─────────────────────────┐
│ Multi-Scale Local       │  Sample anchors, encode neighborhoods
│ Encoder                 │  at multiple radii
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Cross-Point             │  Hilbert curve sampling for
│ Interaction             │  reference points, learn relationships
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Dual-Path Attention     │  Channel + Spatial attention
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Global Aggregation      │  Pool to single feature vector
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Classification Head     │  FC layers with dropout
└─────────────────────────┘
         │
         ▼
    Class Logits [B, C]
```

## Module Structure

- `network.py` - Main classifier network
- `curve_sampling.py` - Hilbert curve sampling implementation
- `feature_layers.py` - Local region encoders and feature upsampling
- `attention.py` - Channel, spatial, and cross-point attention
- `geometry_ops.py` - Point cloud geometric operations

## Key Differences from Similar Works

This implementation makes independent design choices:

1. **Hilbert curves** instead of Morton/Z-order curves (better locality)
2. **Different attention architecture** with separate pathways
3. **Unique module naming and organization**
4. **Independent geometric operation implementations**
5. **Different network composition and data flow**

## Citation

If you use this code, please cite appropriately based on your use case.

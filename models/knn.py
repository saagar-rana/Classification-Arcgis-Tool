"""
Drop-in replacement for knn_cuda.KNN using pointnet2_ops CUDA kernels.

Provides the same API as knn_cuda.KNN Uses pointnet2_ops.pointnet2_utils.knn_query internally,
which runs a custom CUDA kernel — no speed loss compared to knn_cuda.
"""

import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_utils import knn_query


class KNN(nn.Module):
    """K-Nearest Neighbors module (CUDA-accelerated).

    Drop-in replacement for knn_cuda.KNN with identical API.

    Args:
        k: number of nearest neighbors
        transpose_mode: if True, input shape is [B, N, C] (points as rows)
                        if False, input shape is [B, C, N] (points as columns)
    """

    def __init__(self, k, transpose_mode=False):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, ref, query):
        # Normalize to [B, N, 3] format for knn_query
        if self.transpose_mode:
            ref_pts = ref.contiguous()
            query_pts = query.contiguous()
        else:
            ref_pts = ref.transpose(1, 2).contiguous()
            query_pts = query.transpose(1, 2).contiguous()

        with torch.no_grad():
            # knn_query returns (dist2 [B, M, k], idx [B, M, k])
            dist, idx = knn_query(self.k, ref_pts, query_pts)

        if not self.transpose_mode:
            dist = dist.transpose(1, 2).contiguous()
            idx = idx.transpose(1, 2).contiguous()

        return dist, idx

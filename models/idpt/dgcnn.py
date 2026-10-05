"""
IDPT: Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models

Paper: Zha et al., ICCV 2023 — https://arxiv.org/abs/2304.07221
Source: https://github.com/zyh16143998882/ICCV23-IDPT (no license)

Clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dgcnn.edge_conv import EdgeConv  # MIT License — Wang et al. 2019


# Shared helper

def _make_conv1d(in_ch, out_ch, leaky=True):
    act = nn.LeakyReLU(negative_slope=0.2) if leaky else nn.ReLU(inplace=True)
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_ch),
        act,
    )


# Prompt generators.
# Each module takes:
#   x   : [B, C, N]   — patch token features
#   pos : [B, N, C]   — patch positional embeddings (used by transformer only)
# Returns:
#   prompt : [B, 1, C] — a single dynamic prompt token

class PromptGeneratorDeep(nn.Module):
    """
    3-stage EdgeConv prompt generator ('dgcnn' variant in the IDPT paper).
    Applies three graph-conv stages, concatenates them, projects, then
    global-max-pools to a single prompt token.
    """

    def __init__(self, dim: int, k: int = 20, leaky: bool = True):
        super().__init__()
        self.edge1 = EdgeConv(dim, dim, k=k)
        self.edge2 = EdgeConv(dim, dim, k=k)
        self.edge3 = EdgeConv(dim, dim, k=k)
        self.proj = _make_conv1d(dim * 3, dim, leaky=leaky)

    def forward(self, x, pos):
        # x: [B, C, N]
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))           # [B, C, N]
        return F.adaptive_max_pool1d(x, 1).transpose(1, 2)        # [B, 1, C]


class PromptGeneratorMedium(nn.Module):
    """
    2-stage EdgeConv prompt generator ('dgcnn_light2' variant).
    """

    def __init__(self, dim: int, k: int = 20, leaky: bool = True):
        super().__init__()
        self.edge1 = EdgeConv(dim, dim, k=k)
        self.edge2 = EdgeConv(dim, dim, k=k)
        self.proj = _make_conv1d(dim * 2, dim, leaky=leaky)

    def forward(self, x, pos):
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return F.adaptive_max_pool1d(x, 1).transpose(1, 2)


class PromptGeneratorShallow(nn.Module):
    """
    1-stage EdgeConv prompt generator ('dgcnn_light' variant).
    """

    def __init__(self, dim: int, k: int = 20, leaky: bool = True):
        super().__init__()
        self.edge = EdgeConv(dim, dim, k=k)
        self.proj = _make_conv1d(dim, dim, leaky=leaky)

    def forward(self, x, pos):
        x = self.edge(x)
        x = self.proj(x)
        return F.adaptive_max_pool1d(x, 1).transpose(1, 2)


class PromptGeneratorMLP(nn.Module):
    """
    MLP-only prompt generator ('mlp' variant) — no graph convolution.
    """

    def __init__(self, dim: int, leaky: bool = True):
        super().__init__()
        self.proj = _make_conv1d(dim, dim, leaky=leaky)

    def forward(self, x, pos):
        x = self.proj(x)
        return F.adaptive_max_pool1d(x, 1).transpose(1, 2)


class PromptGeneratorTransformer(nn.Module):
    """
    Transformer-based prompt generator ('transformer' variant).
    Runs one self-attention block over the patch tokens and global-max-pools.

    Uses nn.MultiheadAttention (standard PyTorch) with a two-layer MLP,
    following the ViT/Point-BERT block structure (MIT License).
    """

    def __init__(self, dim: int, num_heads: int = 8, drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, pos):
        # x: [B, C, N] → [B, N, C]
        tokens = x.transpose(1, 2) + pos                          # [B, N, C]
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.max(dim=1, keepdim=True)[0]                 # [B, 1, C]

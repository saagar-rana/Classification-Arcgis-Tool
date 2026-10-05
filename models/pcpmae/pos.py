"""
PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders

Paper: NeurIPS 2024
Authors: Xiangdong Zhang, Shaofeng Zhang, Junchi Yan
Source: Implementation adapted from: https://github.com/aHapBean/PCP-MAE
License: MIT
"""
import numpy as np
import torch


def get_pos_embed(embed_dim, ipt_pos):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)
    """
    B, G, C = ipt_pos.size()
    assert embed_dim % 6 == 0
    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device)  # NOTE
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000**omega  # (D/6,)
    rpe = []
    for i in range(C):
        pos_i = ipt_pos[:, :, i]    # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        emb_sin = torch.sin(out)  # (M, D/6)
        emb_cos = torch.cos(out)  # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    return torch.cat(rpe, dim=-1)

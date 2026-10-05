"""
IDPT: Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models

Paper: Zha et al., ICCV 2023 — https://arxiv.org/abs/2304.07221
Source: https://github.com/zyh16143998882/ICCV23-IDPT (no license)

Clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .idpt import Group, _load_pretrained, _init_weights_all
from ..ssl_blocks import Encoder
from ..build import MODELS
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation


# ──────────────────────────────────────────────────────────────
#  DGCNNViewMLP — reference prompt generator
#  Matches ssl/IDPT/segmentation/models/prompt_pt3.py DGCNNViewMLP
# ──────────────────────────────────────────────────────────────

class DGCNNViewMLP(nn.Module):
    """MLP that collapses [B, C, G] → [B, 1, C] via conv + adaptive max-pool."""

    def __init__(self, dim):
        super().__init__()
        self.conv5 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        """x: [B, C, G] → [B, 1, C]."""
        x = self.conv5(x)
        return F.adaptive_max_pool1d(x, 1).transpose(1, 2)  # [B, 1, C]


# ──────────────────────────────────────────────────────────────
#  TransformerPromptEncoderLayerI — exact reference forward
# ──────────────────────────────────────────────────────────────

class TransformerPromptEncoderLayerI(nn.Module):
    """
    Transformer encoder that injects a DGCNN-based dynamic prompt at each
    of three designated layers {3, 7, 11} — exactly as in reference
    prompt_pt3.py TransformerPromptEncoderLayerI.

    At each fetch layer _:
      1. Trim x to last num_group tokens (patch tokens); trim pos too.
      2. Generate prompt via DGCNNViewMLP(patch.T) → [B, 1, C].
      3. Prepend: x ← [prompt | patch], pos ← [prompt_cls_pos | patch_pos].
      4. Run block on extended sequence.
      5. Save [prompt, x_after_block] in feature_list.
         After concat, each entry has shape [B, 2+G, C] because x_after_block
         already contains the injected prompt (1 token) plus the G patch
         tokens; prepending `prompt` a second time adds another token.
      6. Trim x and pos back to patch portion.

    Returns feature_list: list of 3 tensors, each [B, 2+G, C]. The caller
    normalises and transposes to [B, C, 2+G] before slicing (see
    TransformerPromptEncoderLayerIWrapper.forward).
    """

    def __init__(self, embed_dim, depth, num_heads, drop_path_rate, num_group,
                 prompt_layer=11):
        super().__init__()
        from timm.models.layers import DropPath
        from ..ssl_blocks import Block

        self.num_group = num_group
        self.depth = depth
        self.prompt_layer = prompt_layer
        self.fetch_idx = {3, 7, 11}

        dpr = drop_path_rate if isinstance(drop_path_rate, list) else [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.dgcnn_cls3 = DGCNNViewMLP(embed_dim)
        self.dgcnn_cls7 = DGCNNViewMLP(embed_dim)
        self.dgcnn_cls11 = DGCNNViewMLP(embed_dim)
        self._cls_gens = {3: self.dgcnn_cls3, 7: self.dgcnn_cls7, 11: self.dgcnn_cls11}

    def forward(self, x, pos, pos_prompt):
        """
        x          : [B, G, C]  — patch tokens (no cls at start)
        pos        : [B, G, C]  — patch positional embeddings
        pos_prompt : [B, 1, C]  — learnable prompt position

        Returns feature_list: 3 × [B, 2+G, C]
          (original prompt + transformed prompt + G patch tokens; the
          caller applies LayerNorm and transposes to [B, C, 2+G]).
        """
        feature_list = []

        for i, block in enumerate(self.blocks):
            if i in self.fetch_idx:
                # Trim to last num_group patch tokens (reference line verbatim)
                x = x[:, -self.num_group:]
                pos = pos[:, -self.num_group:]

                cls_gen = self._cls_gens[i]
                prompt = cls_gen(x.permute(0, 2, 1))          # [B, 1, C]

                x = torch.cat([prompt, x], dim=1)              # [B, 1+G, C]
                pos = torch.cat([pos_prompt, pos], dim=1)      # [B, 1+G, C]

            x = block(x + pos)

            if i in self.fetch_idx:
                # Save [prompt | x_after_block]: still [B, 1+G, C]
                feature_list.append(torch.cat([prompt, x], dim=1))
                # Restore to patch-only for next layer
                x = x[:, -self.num_group:]
                pos = pos[:, -self.num_group:]

        return feature_list


# ──────────────────────────────────────────────────────────────
#  IDPT_Seg
# ──────────────────────────────────────────────────────────────

@MODELS.register_module()
class IDPT_Seg(BaseSegModel):
    """IDPT part/semantic segmentation.

    Architecture faithfully matches reference prompt_pt3.py get_model.
    Global features are derived from the two prompt-position slices
    ([:,:,:1] and [:,:,1:2]) of each fetched feature, then broadcast
    to N points alongside the label projection.

    Head input dim: 1024 (prop) + 1152 (prompt_feat) + 1152 (prompt_feat2) + 64 (label) = 3392
    """

    FETCH_IDX = {3, 7, 11}

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.get('trans_dim', 384))
        self.depth = int(config.get('depth', 12))
        self.num_heads = int(config.get('num_heads', 6))
        self.num_group = int(config.get('num_group', 128))
        self.group_size = int(config.get('group_size', 32))
        encoder_dims = int(config.get('encoder_dims', self.trans_dim))
        drop_path_rate = float(config.get('drop_path_rate', 0.1))
        dropout = float(config.get('dropout_head', 0.5))

        C = self.trans_dim

        # Tokeniser (same as PointMAE)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        # Learnable prompt position (no cls_token in this seg model)
        self.prompt_cls_pos = nn.Parameter(torch.randn(1, 1, C))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, C),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = TransformerPromptEncoderLayerI(
            embed_dim=C,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=dpr,
            num_group=self.num_group,
            prompt_layer=11,
        )
        self.norm = nn.LayerNorm(C)

        # --- Segmentation head -------------------------------------------
        feat_dim = 3 * C   # 1152 for C=384

        if self.use_cls_label:
            self.label_conv_cls = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64
        else:
            self.label_conv_cls = None
            label_dim = 0

        # prop(1024) + prompt_feature(3C) + prompt_feature_2(3C) + label(64|0)
        self.propagation_0_cls = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[C * 4, 1024],
        )

        in_dim = 1024 + feat_dim + feat_dim + label_dim   # 3392 with label

        self.convs1_cls = nn.Conv1d(in_dim, 512, 1)
        self.bns1_cls = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.convs2_cls = nn.Conv1d(512, 256, 1)
        self.bns2_cls = nn.BatchNorm1d(256)
        self.convs3_cls = nn.Conv1d(256, self.seg_classes, 1)

        trunc_normal_(self.prompt_cls_pos, std=0.02)
        _init_weights_all(self)

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        if ckpt_path is None:
            return
        return _load_pretrained(self, ckpt_path, strict=strict, tag='IDPT_Seg')

    def forward(self, pts, cls_label=None):
        # normalise input
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bcn = pts
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bcn = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"pts must be [B,3,N] or [B,N,3], got {pts.shape}")

        pts_bnc = pts_bcn.transpose(1, 2).contiguous()   # [B, N, 3]
        B, N, _ = pts_bnc.shape

        # --- Tokenise ----------------------------------------------------
        neighborhood, center = self.group_divider(pts_bnc)
        group_input_tokens = self.encoder(neighborhood)    # [B, G, C]

        prompt_cls_pos = self.prompt_cls_pos.expand(B, -1, -1)  # [B, 1, C]
        pos = self.pos_embed(center)                             # [B, G, C]

        # reference: x = group_input_tokens (no cls prepend)
        x = group_input_tokens

        # --- Transformer with prompt injection at {3, 7, 11} -------------
        feature_list = self.blocks(x, pos, prompt_cls_pos)
        # feature_list: 3 × [B, 2+G, C]
        #   index 0 : raw prompt (DGCNN output before block update)
        #   index 1 : transformed prompt (block output at prompt position)
        #   indices 2..(1+G) : patch tokens after block update

        # Apply LayerNorm + transpose to [B, C, 2+G]
        feature_list = [
            self.norm(f).transpose(-1, -2).contiguous()
            for f in feature_list
        ]

        # prompt_feature: first token ([:,:,:1]) from each fetch layer → [B, 3C, N]
        prompt_feature = torch.cat(
            [feature_list[0][:, :, :1],
             feature_list[1][:, :, :1],
             feature_list[2][:, :, :1]], dim=1
        ).expand(-1, -1, N)                                     # [B, 3C, N]

        # prompt_feature_2: second token ([:,:,1:2]) → [B, 3C, N]
        prompt_feature_2 = torch.cat(
            [feature_list[0][:, :, 1:2],
             feature_list[1][:, :, 1:2],
             feature_list[2][:, :, 1:2]], dim=1
        ).expand(-1, -1, N)                                     # [B, 3C, N]

        # x_patch: last num_group tokens from each fetch layer → [B, 3C, G]
        x_patch = torch.cat(
            [feature_list[0][:, :, -self.num_group:],
             feature_list[1][:, :, -self.num_group:],
             feature_list[2][:, :, -self.num_group:]], dim=1
        )                                                        # [B, 3C, G]

        # --- Global feature ----------------------------------------------
        global_parts = [prompt_feature, prompt_feature_2]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_label_feature = self.label_conv_cls(
                cls_label.view(B, self.num_obj_classes, 1)
            ).expand(-1, -1, N)                                  # [B, 64, N]
            global_parts.append(cls_label_feature)
        x_global_feature = torch.cat(global_parts, dim=1)       # [B, 3C+3C+64, N]

        # --- 3-NN FP from G centers to N ---------------------------------
        f_level_0 = self.propagation_0_cls(
            xyz1=pts_bnc,      # [B, N, 3]
            xyz2=center,       # [B, G, 3]
            points1=pts_bcn,   # [B, 3, N]
            points2=x_patch,   # [B, 3C, G]
        )                                                         # [B, 1024, N]

        # --- Head --------------------------------------------------------
        x = torch.cat([f_level_0, x_global_feature], dim=1)      # [B, 3392, N]
        x = F.relu(self.bns1_cls(self.convs1_cls(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2_cls(self.convs2_cls(x)))
        x = self.convs3_cls(x)                                    # [B, seg_classes, N]
        return F.log_softmax(x, dim=1)

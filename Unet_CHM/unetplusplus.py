"""Compact U-Net++ with nested dense skip connections for CHM regression.

Uses one final regression head (no deep supervision) and bilinear upsampling.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convs(x)


class UNetPlusPlusRegressor(nn.Module):
    """Five resolution levels; X[i,j] joins all earlier nodes and X[i+1,j-1]."""

    def __init__(self, in_channels=4, out_channels=1, base_channels=32):
        super().__init__()
        widths = [base_channels * 2 ** level for level in range(5)]
        self.pool = nn.MaxPool2d(2)
        blocks = {}
        for level in range(5):
            blocks[f"x{level}_0"] = DoubleConv(
                in_channels if level == 0 else widths[level - 1], widths[level])
        for column in range(1, 5):
            for level in range(5 - column):
                blocks[f"x{level}_{column}"] = DoubleConv(
                    column * widths[level] + widths[level + 1], widths[level])
        self.blocks = nn.ModuleDict(blocks)
        self.outc = nn.Conv2d(widths[0], out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        nodes = {}
        for level in range(5):
            inputs = x if level == 0 else self.pool(nodes[level - 1, 0])
            nodes[level, 0] = self.blocks[f"x{level}_0"](inputs)
        for column in range(1, 5):
            for level in range(5 - column):
                up = F.interpolate(nodes[level + 1, column - 1],
                                   size=nodes[level, 0].shape[-2:],
                                   mode="bilinear", align_corners=False)
                inputs = [nodes[level, previous] for previous in range(column)] + [up]
                nodes[level, column] = self.blocks[f"x{level}_{column}"](torch.cat(inputs, dim=1))
        return self.relu(self.outc(nodes[0, 4]))

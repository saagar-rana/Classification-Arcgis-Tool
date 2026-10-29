"""Compact residual U-Net for four-band canopy-height regression."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return F.relu(self.convs(x) + self.shortcut(x), inplace=True)


class ResUNetRegressor(nn.Module):
    """Four pooling stages, residual blocks, and a single nonnegative height head."""

    def __init__(self, in_channels=4, out_channels=1, base_channels=32):
        super().__init__()
        widths = [base_channels * 2 ** level for level in range(5)]
        self.encoders = nn.ModuleList([
            ResidualBlock(in_channels if level == 0 else widths[level - 1], width)
            for level, width in enumerate(widths)
        ])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(widths[level + 1], widths[level], 2, stride=2)
            for level in range(3, -1, -1)
        ])
        self.decoders = nn.ModuleList([
            ResidualBlock(2 * widths[level], widths[level])
            for level in range(3, -1, -1)
        ])
        self.outc = nn.Conv2d(widths[0], out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        skips = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x if level == 0 else self.pool(x))
            skips.append(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.relu(self.outc(x))

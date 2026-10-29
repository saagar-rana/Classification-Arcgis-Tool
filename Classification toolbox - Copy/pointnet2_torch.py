"""PointNet++ single-scale grouping using standard PyTorch operations.

Architecture: Qi et al., Deep Hierarchical Feature Learning on Point Sets
in a Metric Space (NeurIPS 2017). No pointnet2_ops extension is required.
"""
from types import SimpleNamespace
import torch
from torch import nn


def gather(points, indices):
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(-1, *([1] * (indices.ndim - 1))).expand_as(indices)
    return points[batch, indices]


@torch.no_grad()
def farthest_points(xyz, count):
    count = min(count, xyz.shape[1])
    indices = torch.empty((xyz.shape[0], count), dtype=torch.long, device=xyz.device)
    distance = torch.full(xyz.shape[:2], float('inf'), device=xyz.device)
    farthest = ((xyz - xyz.mean(1, keepdim=True)) ** 2).sum(-1).argmax(-1)
    batch = torch.arange(xyz.shape[0], device=xyz.device)
    for i in range(count):
        indices[:, i] = farthest
        center = xyz[batch, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - center) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return indices


class SetAbstraction(nn.Module):
    def __init__(self, input_features, widths, count=None, radius=None, neighbors=None):
        super().__init__()
        self.count, self.radius, self.neighbors = count, radius, neighbors
        layers = []
        channels = input_features + 3
        for width in widths:
            layers.extend([nn.Conv2d(channels, width, 1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True)])
            channels = width
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        if self.count is None:
            centers = xyz.mean(1, keepdim=True)
            grouped = torch.cat((xyz.unsqueeze(1) - centers.unsqueeze(2),
                                 features.unsqueeze(1)), dim=-1)
        else:
            with torch.no_grad():
                indices = farthest_points(xyz, self.count)
                centers = gather(xyz, indices)
                distances = torch.cdist(centers.float(), xyz.float())
                values, neighbors = distances.topk(min(self.neighbors, xyz.shape[1]),
                                                   largest=False, sorted=True)
                # Repeat the nearest point where fewer than K lie in the ball.
                neighbors = torch.where(values <= self.radius, neighbors,
                                        neighbors[..., :1].expand_as(neighbors))
            grouped = torch.cat((gather(xyz, neighbors) - centers.unsqueeze(2),
                                 gather(features, neighbors)), dim=-1)
        output = self.mlp(grouped.permute(0, 3, 1, 2)).max(-1).values.transpose(1, 2)
        return centers, output


class PointNet2SSG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.channels_input != 7:
            raise ValueError('PointNet++ requires XYZ, RGB, and intensity (7 features).')
        self.sa1 = SetAbstraction(4, [64, 64, 128], config.npoint1,
                                  config.radius1, config.nsample1)
        self.sa2 = SetAbstraction(128, [128, 128, 256], config.npoint2,
                                  config.radius2, config.nsample2)
        self.sa3 = SetAbstraction(256, [256, 512, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes))

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 7 or points.shape[1] < 1:
            raise ValueError('Expected nonempty point clouds with shape [batch, points, 7].')
        xyz, features = points[..., :3].contiguous(), points[..., 3:].contiguous()
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        _, features = self.sa3(xyz, features)
        return self.classifier(features.squeeze(1))

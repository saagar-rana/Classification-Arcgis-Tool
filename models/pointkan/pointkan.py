"""
PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets

Paper: Computers & Graphics 2025 (arXiv 2410.10084)
Authors: Ali Kashefi
Source: Implementation adapted from: -
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info


class KANshared(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(KANshared, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = torch.tanh(x)

        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2 * i + self.a + self.b - 1) * (2 * i + self.a + self.b) / ((2 * i) * (i + self.a + self.b))
            B = (2 * i + self.a + self.b - 1) * (self.a**2 - self.b**2) / ((2 * i) * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            C = -2 * (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / ((2 * i) * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, :, i] = (A * x + B) * jacobi[:, :, :, i - 1].clone() + C * jacobi[:, :, :, i - 2].clone()

        jacobi = jacobi.permute(0, 2, 3, 1)
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs)
        return y


class KANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(KANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))

        x = torch.tanh(x)

        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2 * i + self.a + self.b - 1) * (2 * i + self.a + self.b) / ((2 * i) * (i + self.a + self.b))
            B = (2 * i + self.a + self.b - 1) * (self.a**2 - self.b**2) / ((2 * i) * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            C = -2 * (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / ((2 * i) * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (A * x + B) * jacobi[:, :, i - 1].clone() + C * jacobi[:, :, i - 2].clone()

        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)
        y = y.view(-1, self.outdim)
        return y


@MODELS.register_module()
class PointKAN(BasePointCloudModel):
    def __init__(self, config, num_classes=40, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointKAN, self).__init__(config, num_classes)

        channels = config.channels_input

        self.poly_degree = config.poly_degree
        self.alpha = config.alpha
        self.beta = config.beta
        self.scale = config.scale

        self.jacobikan5 = KANshared(channels, int(1024 * self.scale), self.poly_degree, self.alpha, self.beta)
        self.jacobikan6 = KANLinear(int(1024 * self.scale), num_classes, self.poly_degree, self.alpha, self.beta)

        self.bn5 = nn.BatchNorm1d(int(1024 * self.scale))

    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] in [3, 6]:
            x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] > 10:
            x = x.permute(0, 2, 1)

        x = self.jacobikan5(x)
        x = self.bn5(x)

        global_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)

        x = self.jacobikan6(global_feature)
        return x

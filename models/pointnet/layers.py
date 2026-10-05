"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

Paper: CVPR 2017
Authors: Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
Source: Implementation adapted from: https://github.com/charlesq34/pointnet
License: MIT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    T-Net: Spatial Transformer Network for PointNet
    Learns a transformation matrix to canonicalize input points

    Args:
        k (int): Dimension of transformation matrix (3 for input transform, 64 for feature transform)
    """

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        # Point-wise convolution layers
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, k * k)

        # Batch normalization for FC layers
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, k, N] where B=batch, k=channels, N=points

        Returns:
            transformation matrix [B, k, k]
        """
        batch_size = x.size(0)

        # Point-wise convolutions with global max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # Add identity matrix to stabilize training
        identity = torch.eye(self.k, dtype=x.dtype, device=x.device)
        identity = identity.view(1, self.k * self.k).expand(batch_size, -1)
        x = x + identity

        # Reshape to transformation matrix
        x = x.view(batch_size, self.k, self.k)

        return x


class FeatureTransformNet(TNet):
    """
    Feature Transformation Network
    Extends T-Net for feature space transformation (64-dimensional)
    """

    def __init__(self, k=64):
        super(FeatureTransformNet, self).__init__(k=k)

    def get_transform_matrix(self):
        """
        Get the current transformation matrix for regularization
        This is used to compute the feature transform regularization loss
        """
        # This would typically be called during forward pass
        # For now, return identity for compatibility
        return torch.eye(self.k).unsqueeze(0)


class PointNetEncoder(nn.Module):
    """
    PointNet encoder that extracts global features from point clouds
    Can be used for both classification and segmentation tasks

    Args:
        input_channels (int): Number of input channels (3 for xyz, 4 for xyz+intensity)
        feature_transform (bool): Whether to use feature transformation
        return_point_features (bool): Whether to return per-point features (for segmentation)
    """

    def __init__(self, input_channels=3, feature_transform=True, return_point_features=False):
        super(PointNetEncoder, self).__init__()

        self.input_channels = input_channels
        self.feature_transform = feature_transform
        self.return_point_features = return_point_features

        # Input transformation
        self.stn = TNet(k=input_channels)

        # First set of convolutions
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transformation
        if feature_transform:
            self.fstn = FeatureTransformNet(k=64)

        # Second set of convolutions
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input point cloud [B, C, N]

        Returns:
            global_feature: Global feature vector [B, 1024]
            point_features: Per-point features [B, 1024, N] (if return_point_features=True)
            trans_input: Input transformation matrix [B, C, C]
            trans_feat: Feature transformation matrix [B, 64, 64] (if feature_transform=True)
        """
        batch_size, _, num_points = x.size()

        # Input transformation
        trans_input = self.stn(x)
        x = x.transpose(2, 1)  # [B, N, C]
        x = torch.bmm(x, trans_input)  # Apply transformation
        x = x.transpose(2, 1)  # Back to [B, C, N]

        # First set of convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)  # [B, N, 64]
            x = torch.bmm(x, trans_feat)  # Apply feature transformation
            x = x.transpose(2, 1)  # Back to [B, 64, N]

        # Store point features before global pooling
        point_features = x.clone()

        # Second set of convolutions
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global max pooling
        global_feature = F.adaptive_max_pool1d(x, 1)
        global_feature = global_feature.view(batch_size, -1)

        if self.return_point_features:
            # Concatenate global feature with point features for segmentation
            global_feature_expanded = global_feature.view(batch_size, -1, 1).expand(-1, -1, num_points)
            point_features = torch.cat([point_features, global_feature_expanded], dim=1)
            return global_feature, point_features, trans_input, trans_feat
        else:
            return global_feature, trans_input, trans_feat


class PointNetClassificationHead(nn.Module):
    """
    Classification head for PointNet

    Args:
        input_dim (int): Input feature dimension (1024 for standard PointNet)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    """

    def __init__(self, input_dim=1024, num_classes=7, dropout=0.5):
        super(PointNetClassificationHead, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Global feature vector [B, input_dim]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def feature_transform_regularizer(trans_feat):
    """
    Compute feature transformation regularization loss
    Encourages the feature transformation matrix to be close to orthogonal

    Args:
        trans_feat: Feature transformation matrix [B, K, K]

    Returns:
        regularization loss
    """
    batch_size, k, _ = trans_feat.size()
    device = trans_feat.device

    # Identity matrix
    identity = torch.eye(k, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    # Compute regularization: ||I - A^T * A||_F^2
    loss = torch.norm(identity - torch.bmm(trans_feat.transpose(2, 1), trans_feat), dim=(1, 2))
    loss = torch.mean(loss)

    return loss * 0.001  # Scale factor as in original paper

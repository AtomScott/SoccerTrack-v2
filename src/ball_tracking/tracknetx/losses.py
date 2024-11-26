import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(WeightedBinaryCrossEntropy, self).__init__()

    def forward(self, y_pred, y):
        """
        Weighted Binary Cross Entropy Loss.

        Args:
            y_pred (torch.Tensor): Predicted outputs.
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = -(
            torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1))
            + torch.square(y_pred)
            * (1 - y)
            * torch.log(torch.clamp(1 - y_pred, 1e-7, 1))
        )
        return torch.mean(loss)


class FocalWeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, gamma=2):
        super(FocalWeightedBinaryCrossEntropy, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y):
        """
        Focal Weighted Binary Cross Entropy Loss.

        Args:
            y_pred (torch.Tensor): Predicted outputs.
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = -(
            torch.square(1 - y_pred)
            * torch.pow(torch.clamp(1 - y_pred, 1e-7, 1), self.gamma)
            * y
            * torch.log(torch.clamp(y_pred, 1e-7, 1))
            + torch.square(y_pred)
            * torch.pow(torch.clamp(y_pred, 1e-7, 1), self.gamma)
            * (1 - y)
            * torch.log(torch.clamp(1 - y_pred, 1e-7, 1))
        )
        return torch.mean(loss)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Dice Loss for binary classification.

        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y):
        """
        Compute Dice Loss.

        Args:
            y_pred (torch.Tensor): Predicted outputs (logits or probabilities).
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Dice loss value.
        """
        y_pred = torch.sigmoid(y_pred)
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)

        intersection = (y_pred_flat * y_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            y_pred_flat.sum() + y_flat.sum() + self.smooth
        )

        return 1 - dice


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Jaccard Loss (IoU Loss) for binary classification.

        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y):
        """
        Compute Jaccard Loss.

        Args:
            y_pred (torch.Tensor): Predicted outputs (logits or probabilities).
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Jaccard loss value.
        """
        y_pred = torch.sigmoid(y_pred)
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)

        intersection = (y_pred_flat * y_flat).sum()
        total = y_pred_flat.sum() + y_flat.sum()
        union = total - intersection

        jaccard = (intersection + self.smooth) / (union + self.smooth)

        return 1 - jaccard


class TotalVariationLoss(nn.Module):
    def __init__(self):
        """
        Total Variation (TV) Loss to encourage spatial smoothness.
        """
        super(TotalVariationLoss, self).__init__()

    def forward(self, inputs):
        """
        Compute Total Variation Loss.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: TV loss value.
        """
        batch_size = inputs.size(0)
        h_tv = torch.mean(torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1]))
        return (h_tv + w_tv) / batch_size


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        Edge Loss to focus on boundaries within the heatmaps.
        """
        super(EdgeLoss, self).__init__()
        # Define Sobel filters
        sobel_kernel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.register_buffer("sobel_x", sobel_kernel_x)
        self.register_buffer("sobel_y", sobel_kernel_y)

    def forward(self, inputs, targets):
        """
        Compute Edge Loss.

        Args:
            inputs (torch.Tensor): Predicted outputs (logits or probabilities).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Edge loss value.
        """
        # Apply Sobel filters to compute gradients
        inputs_edge_x = F.conv2d(inputs, self.sobel_x, padding=1)
        inputs_edge_y = F.conv2d(inputs, self.sobel_y, padding=1)
        targets_edge_x = F.conv2d(targets, self.sobel_x, padding=1)
        targets_edge_y = F.conv2d(targets, self.sobel_y, padding=1)

        # Compute L1 loss on edges
        edge_loss = F.l1_loss(inputs_edge_x, targets_edge_x) + F.l1_loss(
            inputs_edge_y, targets_edge_y
        )
        return edge_loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        """
        Perceptual Loss using a pretrained VGG network to capture high-level differences.
        """
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, inputs, targets):
        """
        Compute Perceptual Loss.

        Args:
            inputs (torch.Tensor): Predicted outputs (logits or probabilities).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Perceptual loss value.
        """
        # Upscale inputs to match VGG input size if necessary
        inputs = F.interpolate(
            inputs, size=(224, 224), mode="bilinear", align_corners=False
        )
        targets = F.interpolate(
            targets, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Convert to 3 channels if necessary
        if inputs.size(1) != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)

        # Normalize inputs to VGG's expected input range
        mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        targets = (targets - mean) / std

        # Extract features
        inputs_features = self.features(inputs)
        targets_features = self.features(targets)

        # Compute L1 loss between features
        return self.criterion(inputs_features, targets_features)

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
            + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1))
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
        dice = (2.0 * intersection + self.smooth) / (y_pred_flat.sum() + y_flat.sum() + self.smooth)

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
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        sobel_kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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
        edge_loss = F.l1_loss(inputs_edge_x, targets_edge_x) + F.l1_loss(inputs_edge_y, targets_edge_y)
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
        inputs = F.interpolate(inputs, size=(224, 224), mode="bilinear", align_corners=False)
        targets = F.interpolate(targets, size=(224, 224), mode="bilinear", align_corners=False)

        # Convert to 3 channels if necessary
        if inputs.size(1) != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)

        # Normalize inputs to VGG's expected input range
        mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        targets = (targets - mean) / std

        # Extract features
        inputs_features = self.features(inputs)
        targets_features = self.features(targets)

        # Compute L1 loss between features
        return self.criterion(inputs_features, targets_features)


class AwingLoss(nn.Module):
    """
    Awing loss for heatmap regression, adapted from:
    Feng et al. "Wing Loss for Robust Facial Landmark Localization with
    Convolutional Neural Networks." CVPR 2018.

    The loss is defined for x = difference between predicted and target values.
    If |x| < w: loss = c * log(1 + (|x|/epsilon)^alpha)
    If |x| >= w: loss = |x| - c

    Typical parameters from the paper:
    w=10, epsilon=2, alpha=2.1
    """

    def __init__(self, w=10.0, epsilon=2.0, alpha=2.1):
        super(AwingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.alpha = alpha
        # Compute c constant
        self.c = self.w - self.w * (1.0 / (1 + (self.w / self.epsilon) ** self.alpha))

    def forward(self, pred, target):
        # pred, target: shape (B, C, H, W)
        x = target - pred
        abs_x = torch.abs(x)
        mask_small = abs_x < self.w
        mask_large = ~mask_small

        loss_small = self.c * torch.log(1 + (abs_x[mask_small] / self.epsilon) ** self.alpha)
        loss_large = abs_x[mask_large] - self.c
        return (loss_small.sum() + loss_large.sum()) / (pred.numel())


class WeightedMSELoss(nn.Module):
    def __init__(self, foreground_weight=1.0, background_weight=0.1, threshold=0.01):
        """
        Weighted MSE Loss.
        Args:
            foreground_weight (float): Weight for pixels considered 'foreground'.
            background_weight (float): Weight for pixels considered 'background'.
            threshold (float): A value to determine what counts as foreground.
        """
        super(WeightedMSELoss, self).__init__()
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.threshold = threshold

    def forward(self, y_pred, y):
        """
        Args:
            y_pred (torch.Tensor): Predicted heatmaps (B, C, H, W), probabilities or logits.
            y (torch.Tensor): Ground truth heatmaps (B, C, H, W), with Gaussian peaks.

        Returns:
            torch.Tensor: Weighted MSE loss value.
        """
        # If y is a Gaussian heatmap with small positive values around the ball,
        # we can consider any pixel above a certain threshold as 'foreground'.
        foreground_mask = y > self.threshold

        # Apply different weights
        weights = torch.ones_like(y)
        weights[foreground_mask] = self.foreground_weight
        weights[~foreground_mask] = self.background_weight

        loss = ((y_pred - y) ** 2 * weights).mean()
        return loss


class CoordinateHeatmapLoss(nn.Module):
    """
    Combined loss for coordinate regression + heatmap quality.

    Steps:
    1. Compute predicted coordinates via soft-argmax on predicted heatmaps.
    2. Use Smooth-L1 loss to supervise coordinates.
    3. Use Awing loss to supervise heatmaps.
    4. Combine them with a weighting factor beta.

    Args:
        beta (float): Weight for the Awing (heatmap) component relative to coordinate loss.
    """

    def __init__(self, beta=0.1):
        super(CoordinateHeatmapLoss, self).__init__()
        self.beta = beta
        self.coordinate_loss_fn = nn.SmoothL1Loss()
        self.heatmap_loss_fn = AwingLoss()  # Or any other robust heatmap loss

    def forward(self, pred_heatmaps, gt_heatmaps, gt_coords):
        """
        Args:
            pred_heatmaps (torch.Tensor): Predicted heatmaps of shape (B, C, H, W).
            gt_heatmaps   (torch.Tensor): Ground truth heatmaps of shape (B, C, H, W).
            gt_coords     (torch.Tensor): Ground truth coordinates of shape (B, C, 2).

        Returns:
            torch.Tensor: Combined loss value.
        """

        # pred_heatmaps are logits or probabilities. We assume probabilities here:
        # Ensure pred_heatmaps are probabilities (0-1)
        # If model outputs logits, apply sigmoid:
        # (In the model code, you apply torch.sigmoid() before calling this loss, so pred_heatmaps should already be probabilities.)

        B, C, H, W = pred_heatmaps.shape

        # 1. Compute predicted coordinates via soft-argmax
        # Soft-argmax along height and width for each channel
        # pred_heatmaps: B x C x H x W
        # We can compute a probability distribution per channel and then
        # weighted sum of pixel indices gives coordinates.

        # Flatten height & width
        heatmaps_flat = pred_heatmaps.view(B, C, -1)  # shape: B x C x (H*W)
        # Normalize each heatmap to sum to 1
        heatmaps_flat = F.softmax(heatmaps_flat, dim=2)  # B x C x (H*W)

        # Compute coordinates
        # Create a grid of coordinate indices
        y_coords = torch.arange(H, device=pred_heatmaps.device).view(1, 1, H, 1).repeat(B, C, 1, W)
        x_coords = torch.arange(W, device=pred_heatmaps.device).view(1, 1, 1, W).repeat(B, C, H, 1)

        y_coords_flat = y_coords.view(B, C, -1).float()
        x_coords_flat = x_coords.view(B, C, -1).float()

        pred_x = torch.sum(x_coords_flat * heatmaps_flat, dim=2)  # B x C
        pred_y = torch.sum(y_coords_flat * heatmaps_flat, dim=2)  # B x C

        pred_coords = torch.stack([pred_x, pred_y], dim=2)  # B x C x 2

        # 2. Coordinate loss (Smooth-L1)
        coord_loss = self.coordinate_loss_fn(pred_coords, gt_coords)

        # 3. Heatmap loss (Awing)
        heatmap_loss = self.heatmap_loss_fn(pred_heatmaps, gt_heatmaps)

        # 4. Combine losses
        total_loss = coord_loss + self.beta * heatmap_loss
        return total_loss
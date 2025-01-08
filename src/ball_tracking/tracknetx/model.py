import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb  # Added for W&B Image logging

# Import loss functions
from src.ball_tracking.tracknetx.losses import (
    WeightedBinaryCrossEntropy,
    FocalWeightedBinaryCrossEntropy,
    DiceLoss,
    JaccardLoss,
    TotalVariationLoss,
    EdgeLoss,
    PerceptualLoss,
    CoordinateHeatmapLoss,
    WeightedMSELoss,
)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        """
        Channel Attention Module as part of CBAM.

        Args:
            channel (int): Number of input channels.
            ratio (int): Reduction ratio for hidden layer.
        """
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Channel Attention Module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, height, width).

        Returns:
            torch.Tensor: Output tensor after applying channel attention.
        """
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        """
        Spatial Attention Module as part of CBAM.
        """
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Spatial Attention Module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, height, width).

        Returns:
            torch.Tensor: Output tensor after applying spatial attention.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        scale = self.sigmoid(out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channel):
        """
        Convolutional Block Attention Module (CBAM).

        Args:
            channel (int): Number of input channels.
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        """
        Forward pass for CBAM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying CBAM.
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Conv2DBlock(nn.Module):
    """
    Convolutional Block: Conv2D -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding="same", bias=True):
        """
        Initialize Conv2DBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolving kernel.
            padding (str or int): Padding added to all four sides of the input.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super(Conv2DBlock, self).__init__()
        if isinstance(kernel_size, tuple):
            padding_val = kernel_size[0] // 2 if padding == "same" else padding
        else:
            padding_val = kernel_size // 2 if padding == "same" else padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_val,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for Conv2DBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionBlock(nn.Module):
    """
    Inception-like Block using multiple convolutional paths.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize InceptionBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for each path.
        """
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            Conv2DBlock(in_channels, out_channels, kernel_size=(1, 1)),
            Conv2DBlock(out_channels, out_channels, kernel_size=(3, 3)),
        )
        self.branch2 = nn.Sequential(
            Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3)),
            Conv2DBlock(out_channels, out_channels, kernel_size=(3, 3)),
        )
        self.branch3 = nn.Sequential(
            Conv2DBlock(in_channels, out_channels, kernel_size=(5, 5)),
            Conv2DBlock(out_channels, out_channels, kernel_size=(3, 3)),
        )
        self.conv_out = Conv2DBlock(out_channels * 3, out_channels, kernel_size=(3, 3))

    def forward(self, x):
        """
        Forward pass for InceptionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        out = self.conv_out(x_cat)
        return out + x2  # Residual connection


class TripleConv(nn.Module):
    """
    Triple convolutional block: Conv2DBlock x 3
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize TripleConv.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(TripleConv, self).__init__()
        self.conv1 = Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3))
        self.conv2 = Conv2DBlock(out_channels, out_channels, kernel_size=(3, 3))
        self.conv3 = Conv2DBlock(out_channels, out_channels, kernel_size=(3, 3))

    def forward(self, x):
        """
        Forward pass for TripleConv.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DoubleConv(nn.Module):
    """
    Double convolutional block: Conv2DBlock x 2
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize DoubleConv.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.conv1 = Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3))
        self.conv2 = Conv2DBlock(out_channels, out_channels, kernel_size=(3, 3))

    def forward(self, x):
        """
        Forward pass for DoubleConv.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TrackNetXModel(pl.LightningModule):
    """
    PyTorch Lightning Module for the TrackNetX model.
    """

    def __init__(
        self,
        in_channels=9,
        out_channels=3,
        learning_rate=0.001,
        loss_function="weighted_bce",
        aux_loss_functions=None,
        aux_loss_weights=None,
        **kwargs,
    ):
        """
        Initialize the TrackNetXModel.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            learning_rate (float): Learning rate for the optimizer.
            loss_function (str): Main loss function to use.
            aux_loss_functions (list, optional): List of auxiliary loss function names.
            aux_loss_weights (list, optional): List of weights for auxiliary loss functions.
        """
        super(TrackNetXModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define layers
        self.down_block_1 = InceptionBlock(in_channels, 64)
        self.down_block_2 = InceptionBlock(64, 128)
        self.down_block_3 = InceptionBlock(128, 256)
        self.bottleneck = TripleConv(256, 512)
        self.up_block_1 = DoubleConv(768, 256)
        self.up_block_2 = DoubleConv(384, 128)
        self.up_block_3 = DoubleConv(192, 64)
        self.predictor = nn.Conv2d(64, out_channels, kernel_size=(1, 1))

        # Attention modules
        self.cbam1 = CBAM(channel=256)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=64)
        self.cbam1_2 = CBAM(channel=256)
        self.cbam2_2 = CBAM(channel=128)
        self.cbam3_2 = CBAM(channel=64)

        # Initialize the main loss function
        if loss_function == "weighted_bce":
            self.main_criterion = WeightedBinaryCrossEntropy()
        elif loss_function == "focal_wbce":
            self.main_criterion = FocalWeightedBinaryCrossEntropy(gamma=2)
        elif loss_function == "bce":
            self.main_criterion = nn.BCELoss()
        elif loss_function == "coordinate_heatmap":
            self.main_criterion = CoordinateHeatmapLoss()
        elif loss_function == "weighted_mse":
            self.main_criterion = WeightedMSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # Initialize auxiliary loss functions
        self.aux_criteria = nn.ModuleList()
        self.aux_weights = []

        if aux_loss_functions and aux_loss_weights:
            if len(aux_loss_functions) != len(aux_loss_weights):
                raise ValueError("aux_loss_functions and aux_loss_weights must have the same length.")
            for aux_loss_name, aux_weight in zip(aux_loss_functions, aux_loss_weights):
                if aux_loss_name == "dice":
                    aux_loss = DiceLoss()
                elif aux_loss_name == "jaccard":
                    aux_loss = JaccardLoss()
                elif aux_loss_name == "tv":
                    aux_loss = TotalVariationLoss()
                elif aux_loss_name == "edge":
                    aux_loss = EdgeLoss()
                elif aux_loss_name == "perceptual":
                    aux_loss = PerceptualLoss()
                else:
                    raise ValueError(f"Unsupported auxiliary loss function: {aux_loss_name}")
                self.aux_criteria.append(aux_loss)
                self.aux_weights.append(aux_weight)
        elif aux_loss_functions or aux_loss_weights:
            raise ValueError("Both aux_loss_functions and aux_loss_weights must be provided together.")

    def forward(self, x):
        """
        Forward pass of the TrackNetX model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor.
        """
        # Down-sampling path
        x1 = self.down_block_1(x)
        x = F.max_pool2d(x1, kernel_size=2)
        x2 = self.down_block_2(x)
        x = F.max_pool2d(x2, kernel_size=2)
        x3 = self.down_block_3(x)
        x = F.max_pool2d(x3, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Attention and skip connections
        x3 = self.cbam1_2(x3)
        x = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)

        # Up-sampling path
        x = self.up_block_1(x)
        x = self.cbam1(x)
        x2 = self.cbam2_2(x2)
        x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)

        x = self.up_block_2(x)
        x = self.cbam2(x)
        x1 = self.cbam3_2(x1)
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)

        x = self.up_block_3(x)
        x = self.cbam3(x)

        # Output layer
        x = self.predictor(x)
        return x  # Return logits

    def _compute_and_log_losses(self, outputs, outputs_probs, heatmaps, coords, prefix="train"):
        """
        Compute and log main loss, auxiliary losses, and total loss.

        Args:
            outputs (torch.Tensor): Model output logits
            outputs_probs (torch.Tensor): Sigmoid of model outputs
            heatmaps (torch.Tensor): Ground truth heatmaps
            prefix (str): Prefix for logging ("train" or "val")

        Returns:
            torch.Tensor: Total loss
        """
        # If using coordinate_heatmap loss:
        if isinstance(self.main_criterion, CoordinateHeatmapLoss):
            # main loss = Smooth-L1(coord) + beta * Awing(heatmap)
            main_loss = self.main_criterion(outputs_probs, heatmaps, coords)
        else:
            # Original code if not using coordinate_heatmap
            main_loss = self.main_criterion(outputs_probs, heatmaps)

        self.log(f"{prefix}_main_loss", main_loss, prog_bar=True)

        total_loss = main_loss
        aux_loss_sum = 0

        # Compute auxiliary losses
        if self.aux_criteria:
            for aux_loss_fn, weight in zip(self.aux_criteria, self.aux_weights):
                # Handle losses that may or may not require targets
                try:
                    # Try with both inputs and targets
                    aux_loss_value = aux_loss_fn(outputs_probs, heatmaps)
                except TypeError:
                    # If targets are not required
                    aux_loss_value = aux_loss_fn(outputs_probs)

                weighted_aux_loss = weight * aux_loss_value
                aux_loss_sum += weighted_aux_loss

                # Log individual auxiliary loss
                self.log(f"{prefix}_{aux_loss_fn.__class__.__name__}", aux_loss_value, prog_bar=False)

            # Log sum of auxiliary losses
            self.log(f"{prefix}_aux_loss", aux_loss_sum, prog_bar=True)
            total_loss = main_loss + aux_loss_sum

        # Log total loss
        self.log(f"{prefix}_total_loss", total_loss, prog_bar=True)

        # Get coordinates of peak in heatmaps
        batch_size, num_maps, height, width = outputs_probs.shape

        # Flatten spatial dimensions and find max index for all maps
        flat_indices = outputs_probs.reshape(batch_size, num_maps, -1).argmax(dim=2)  # Shape: (batch_size, num_maps)

        # Convert flat indices to 2D coordinates
        pred_y = flat_indices // width
        pred_x = flat_indices % width

        # Stack x,y coordinates - Shape: (batch_size, num_maps, 2)
        pred_coords = torch.stack([pred_x, pred_y], dim=2).float()
        dist = torch.norm(coords - pred_coords, dim=2).mean()
        self.log(f"{prefix}_dist", dist, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        _, frames, heatmaps, coords = batch
        outputs = self(frames)
        outputs_probs = torch.sigmoid(outputs)

        # Compute and log losses without batch_idx
        total_loss = self._compute_and_log_losses(outputs, outputs_probs, heatmaps, coords, prefix="train")

        if batch_idx == 0:
            grid = self.create_grid(frames, heatmaps, outputs_probs)
            self.logger.experiment.log(
                {
                    "Train_Results": [wandb.Image(grid)],
                    "epoch": self.current_epoch,
                }
            )
        return total_loss

    def validation_step(self, batch, batch_idx):
        _, frames, heatmaps, coords = batch
        outputs = self(frames)
        outputs_probs = torch.sigmoid(outputs)

        # Compute and log losses without batch_idx
        self._compute_and_log_losses(outputs, outputs_probs, heatmaps, coords, prefix="val")

        # Periodic visualization
        if batch_idx == 0:
            grid = self.create_grid(frames, heatmaps, outputs_probs)
            self.logger.experiment.log(
                {
                    "Validation_Results": [wandb.Image(grid)],
                    "epoch": self.current_epoch,
                }
            )

    def test_step(self, batch, batch_idx):
        _, frames, heatmaps, coords = batch
        outputs = self(frames)
        outputs_probs = torch.sigmoid(outputs)

        # Compute and log losses without batch_idx
        self._compute_and_log_losses(outputs, outputs_probs, heatmaps, coords, prefix="test")

    def create_grid(self, frames, heatmaps, outputs):
        """
        Create a grid of images for visualization.

        Args:
            frames (torch.Tensor): Input frames of shape (batch_size, num_input_frames * 3, height, width)
            heatmaps (torch.Tensor): Ground truth heatmaps of shape (batch_size, 3, height, width)
            outputs (torch.Tensor): Predicted heatmaps of shape (batch_size, 3, height, width)

        Returns:
            numpy.ndarray: Grid image in NumPy format.
        """
        import torchvision

        # Select first sample in the batch and get middle frame
        frame = frames[0].detach().cpu()  # Shape: (9, height, width)
        middle_frame = frame[3:6]  # Select middle RGB frame
        gt = heatmaps[0, 1].detach().cpu()  # Select middle heatmap channel
        pred = outputs[0, 1].detach().cpu()  # Select middle heatmap channel

        # Normalize for visualization
        try:
            middle_frame = (middle_frame - middle_frame.min()) / (middle_frame.max() - middle_frame.min())
        except:
            middle_frame = torch.zeros_like(middle_frame)
        try:
            gt = (gt - gt.min()) / (gt.max() - gt.min())
        except:
            gt = torch.zeros_like(gt)
        try:
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        except:
            pred = torch.zeros_like(pred)

        # Create colored heatmap overlays (red for ground truth, blue for prediction)
        gt_colored = torch.zeros((3, *gt.shape), device=gt.device)
        gt_colored[0] = gt  # Red channel

        pred_colored = torch.zeros((3, *pred.shape), device=pred.device)
        pred_colored[0] = pred  # Red channel

        # Combine images with alpha blending
        alpha = 0.5  # Adjust overlay transparency
        gt_viz = middle_frame * (1 - alpha) + gt_colored * alpha
        pred_viz = middle_frame * (1 - alpha) + pred_colored * alpha

        # Combine all visualizations
        images = torch.stack([gt_viz, pred_viz], dim=0)
        grid = torchvision.utils.make_grid(images, nrow=1)

        # Convert to numpy and transpose from CHW to HWC format
        grid = grid.numpy().transpose(1, 2, 0)

        return grid

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

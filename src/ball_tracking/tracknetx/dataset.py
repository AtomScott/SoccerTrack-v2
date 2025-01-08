import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from loguru import logger


class TrackNetX_Dataset(Dataset):
    def __init__(
        self,
        frame_files,
        coordinates,
        visibility,
        num_frame=3,
        mag=1,
        sigma=2.5,
        augmentations=None,
    ):
        """
        PyTorch Dataset for TrackNetX ball tracking. This dataset handles sequences of frames
        and their corresponding ball coordinates to generate training data for the TrackNetX model.

        The dataset processes sequences of frames and generates Gaussian-like heatmaps centered on the
        ball coordinates. Each sample consists of a sequence of frames and their corresponding heatmaps.

        The major difference from the original version is that we now use a precomputed kernel for the
        heatmap. Instead of computing a full-frame distance map for every coordinate, we focus only on
        a small patch around the ball location. This significantly reduces computation time, especially
        for large images.

        Args:
            frame_files (np.ndarray): Array of frame file paths. Shape (N, num_frame) where N is the
                number of sequences and num_frame is the number of consecutive frames per sequence.
            coordinates (np.ndarray): Ball coordinates for each frame. Shape (N, num_frame, 2) where
                the last dimension contains [x, y] pixel coordinates.
            visibility (np.ndarray): Binary visibility flags indicating if the ball is visible in each frame.
                Shape (N, num_frame) where 1 indicates visible and 0 indicates not visible.
            num_frame (int, optional): Number of consecutive frames in each sequence. Defaults to 3.
            mag (float, optional): Magnitude of the heatmap peak. Defaults to 1.
            sigma (float, optional): Standard deviation (radius) of the circular region around the ball.
                Pixels within sigma radius are set to 'mag', others are set to 0. Defaults to 2.5.
            augmentations (callable, optional): A function/transform that takes in frames, heatmaps,
                and coordinates and returns their transformed versions. Defaults to None.
        """
        self.frame_files = frame_files
        self.coordinates = coordinates
        self.visibility = visibility
        self.num_frame = num_frame
        self.mag = mag
        self.sigma = sigma
        self.augmentations = augmentations

        # Read a sample image to determine the height and width of frames
        sample_img = cv2.imread(self.frame_files[0][0])
        if sample_img is None:
            raise ValueError(f"Failed to load sample image: {self.frame_files[0][0]}")
        self.h, self.w, _ = sample_img.shape

        # Preallocate the frames array template
        self.frames_template = np.empty((num_frame * 3, self.h, self.w), dtype=np.float32)

        # Precompute the kernel (a small circular patch) once
        self.kernel_radius = int(np.ceil(self.sigma))
        x = np.arange(-self.kernel_radius, self.kernel_radius + 1)
        y = np.arange(-self.kernel_radius, self.kernel_radius + 1)
        X, Y = np.meshgrid(x, y)
        dist_sq = X**2 + Y**2
        self.kernel = np.where(dist_sq <= self.sigma**2, self.mag, 0.0).astype(np.float32)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return len(self.frame_files)

    def __getitem__(self, idx):
        """
        Retrieves a single training sample consisting of a sequence of frames and their heatmaps.

        Steps:
        1. Loads the sequence of frames from disk.
        2. Converts frames from BGR to RGB and normalizes to [0, 1].
        3. Generates heatmaps for visible ball locations using a precomputed kernel.
        4. Applies any specified augmentations.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            tuple:
                - idx (int): The index of the retrieved sequence.
                - frames (np.ndarray): Concatenated RGB frames normalized to [0, 1].
                  Shape: (num_frame * 3, H, W) where H, W are the frame dimensions.
                - heatmaps (np.ndarray): Generated heatmaps. Shape: (num_frame, H, W).
                - coors (np.ndarray): Ball coordinates for each frame (num_frame, 2).
        """
        frame_file = self.frame_files[idx]
        coors = self.coordinates[idx].copy()
        vis = self.visibility[idx]

        # Use preallocated array template
        frames = self.frames_template.copy()  # Create a copy of the template

        # Load and process frames directly into preallocated array
        for i in range(self.num_frame):
            # Read image
            img = cv2.imread(frame_file[i])
            if img is None:
                raise ValueError(f"Failed to load image: {frame_file[i]}")

            # Process image and write directly to frames array
            # Note: astype and normalization done in one step
            frames[i * 3 : (i + 1) * 3] = (
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
            ) / 255.0

        # Generate corresponding heatmaps
        heatmaps = self._generate_heatmaps(coors, vis)

        logger.debug(f"Sample frame data for idx {idx}: shape={frames.shape}")
        logger.debug(f"Sample heatmap data for idx {idx}: shape={heatmaps.shape}")

        # Apply augmentations if any
        if self.augmentations:
            frames, heatmaps, coors = self.augmentations(frames, heatmaps, coors)

        return idx, frames, heatmaps, coors

    def _generate_heatmaps(self, coors, vis):
        """
        Generates heatmaps for a sequence of ball coordinates.

        For each frame where the ball is visible (vis[i] == 1), we create a heatmap by placing a
        precomputed circular kernel (defined by sigma and mag) at the ball's location. This avoids
        the expensive computation of a full-frame Gaussian for every frame.

        Args:
            coors (np.ndarray): Ball coordinates for each frame. Shape: (num_frame, 2).
            vis (np.ndarray): Visibility flags for each frame. Shape: (num_frame).

        Returns:
            np.ndarray: Heatmaps with shape (num_frame, h, w). Each heatmap has a small circular
            region of 'mag' values at the ball location, and 0 elsewhere.
        """
        heatmaps = np.zeros((self.num_frame, self.h, self.w), dtype=np.float32)
        for i in range(self.num_frame):
            if vis[i]:  # Only generate heatmap if the object is visible
                heatmap = self._get_heatmap(int(coors[i][0]), int(coors[i][1]))
                heatmaps[i] = heatmap
        return heatmaps

    def _get_heatmap(self, cx, cy):
        """
        Generates a single heatmap for given ball coordinates by "stamping" the precomputed kernel
        at the ball's location.

        We compute only a small patch around (cx, cy), defined by kernel_radius. The patch is placed
        into a zero array of shape (h, w).

        Args:
            cx (int): X-coordinate of the ball center in pixels.
            cy (int): Y-coordinate of the ball center in pixels.

        Returns:
            np.ndarray: Generated heatmap with shape (h, w). Values are 'mag' inside a radius defined
            by sigma, and 0 outside that radius.
        """
        heatmap = np.zeros((self.h, self.w), dtype=np.float32)

        # Determine region of interest
        x_min = max(cx - self.kernel_radius, 0)
        x_max = min(cx + self.kernel_radius, self.w - 1)
        y_min = max(cy - self.kernel_radius, 0)
        y_max = min(cy + self.kernel_radius, self.h - 1)

        # Coordinates in kernel
        kernel_x_start = self.kernel_radius - (cx - x_min)
        kernel_y_start = self.kernel_radius - (cy - y_min)
        kernel_x_end = kernel_x_start + (x_max - x_min + 1)
        kernel_y_end = kernel_y_start + (y_max - y_min + 1)

        # Place the kernel patch into the heatmap
        heatmap[y_min : y_max + 1, x_min : x_max + 1] = self.kernel[
            kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end
        ]

        return heatmap


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from pathlib import Path
    import matplotlib.pyplot as plt
    import sys

    # Load config
    config_path = Path("configs/default_config.yaml")
    cfg = OmegaConf.load(config_path)
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Get dataset paths from config
    dataset_dir = Path(cfg.ball_tracking.data.output_dir)
    train_frames_path = dataset_dir / "train/sequences.npy"
    train_coords_path = dataset_dir / "train/coordinates.npy"
    train_vis_path = dataset_dir / "train/visibility.npy"

    # Load the dataset files
    frame_files = np.load(train_frames_path)
    coordinates = np.load(train_coords_path)
    visibility = np.load(train_vis_path)

    logger.info(f"Loaded dataset with {len(frame_files)} sequences")
    logger.info(f"Frame files shape: {frame_files.shape}")
    logger.info(f"Coordinates shape: {coordinates.shape}")
    logger.info(f"Visibility shape: {visibility.shape}")

    # Create dataset instance
    dataset = TrackNetX_Dataset(
        frame_files=frame_files,
        coordinates=coordinates,
        visibility=visibility,
        num_frame=cfg.ball_tracking.data.num_frame,
        mag=cfg.ball_tracking.data.mag,
        sigma=cfg.ball_tracking.data.sigma,
    )

    # time __getitem__
    import timeit

    def load_image(frame_file):
        for fp in frame_file:
            cv2.imread(fp)

    logger.info(f"Time taken for __getitem__: {timeit.timeit(lambda: dataset[0], number=10) / 10} seconds")

    # Display samples vertically with overlaid heatmaps
    sample_idx = 10
    num_frames = 3
    fig = plt.figure(figsize=(10, 8))

    idx, frames, heatmaps, coords = dataset[sample_idx]

    logger.info(f"Frames shape: {frames.shape}")
    logger.info(f"Heatmaps shape: {heatmaps.shape}")
    logger.info(f"Coords shape: {coords.shape}")
    # For each frame in the sequence
    for i in range(num_frames):  # 3 frames per sequence
        plt.subplot(num_frames, 1, i + 1)

        # Get the RGB frame
        frame = frames[i * 3 : (i + 1) * 3].transpose(1, 2, 0)  # Convert to HWC

        # Create overlay
        cv2.imwrite(f"frame_{i}.png", frame * 255)
        plt.imshow(frame, interpolation="nearest")

        # Overlay heatmap with alpha blending
        # heatmap_overlay = plt.imshow(heatmaps[i], cmap="hot", alpha=0.5, interpolation="nearest")

        # Add colorbar with fixed range from 0 to 1
        # plt.colorbar(heatmap_overlay, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0])

        # Mark the actual ball position if visible
        if coords[i][0] != 0 and coords[i][1] != 0:  # If ball is visible
            plt.plot(coords[i][0], coords[i][1], "ro", mfc="none", mec="red", markersize=12, label="Ball Position")

        plt.title(f"Sample {sample_idx}, Frame {i}")
        plt.axis("on")  # Show axes to help with coordinate interpretation

    # Save each sample to a separate file
    plt.savefig("dataset_visualization.jpg", bbox_inches="tight", dpi=300)
    logger.info(f"Saved to: {Path.cwd() / 'dataset_visualization.jpg'}")

    # Print some statistics
    logger.info("Dataset Statistics:")
    logger.info(f"Total sequences: {len(dataset)}")
    logger.info(f"Frame dimensions: {frames.shape}")
    logger.info(f"Heatmap dimensions: {heatmaps.shape}")

    # Check visibility distribution
    vis_ratio = np.mean(visibility)
    logger.info(f"Average visibility ratio: {vis_ratio:.2f}")

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
        PyTorch Dataset for the TrackNetX tracking data.

        Args:
            frame_files (np.ndarray): Array of frame file paths with shape (N, num_frame).
            coordinates (np.ndarray): Array of coordinates with shape (N, num_frame, 2).
            visibility (np.ndarray): Array of visibility flags with shape (N, num_frame).
            num_frame (int): Number of frames in each sample.
            mag (float): Magnification factor for heatmap generation.
            sigma (float): Sigma value for heatmap generation (controls the spread of the heatmap).
            augmentations (callable, optional): A function/transform to apply to the data.
        """
        self.frame_files = frame_files  # Shape: (N, num_frame)
        self.coordinates = coordinates  # Shape: (N, num_frame, 2)
        self.visibility = visibility  # Shape: (N, num_frame)

        self.num_frame = num_frame
        self.mag = mag
        self.sigma = sigma

        self.augmentations = augmentations  # Assign augmentations

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Returns:
            idx (int): Index of the sample.
            frames (np.ndarray): Concatenated frames of shape (num_frame * 3, H, W).
            heatmaps (np.ndarray): Heatmaps corresponding to frames, shape (num_frame, H, W).
            coors (np.ndarray): Coordinates of the object in the frames, shape (num_frame, 2).
        """
        frame_file = self.frame_files[idx]
        coors = self.coordinates[idx].copy()  # Copy to avoid modifying original data
        vis = self.visibility[idx]

        # Verify frame existence
        if not all(os.path.exists(fp) for fp in frame_file):
            raise FileNotFoundError(f"Missing frame files for idx {idx}")

        # Read first frame to get original dimensions
        img = cv2.imread(frame_file[0])
        if img is None:
            raise ValueError(f"Failed to load image: {frame_file[0]}")

        h, w, _ = img.shape

        # Log debug information
        logger.debug(f"Frame file paths for idx {idx}: {frame_file}")
        logger.debug(f"Original coordinates for idx {idx}: {coors}")
        logger.debug(f"Visibility for idx {idx}: {vis}")
        logger.debug(f"Original dimensions: height={h}, width={w}")

        # Read and process frames
        frames = []
        for i in range(self.num_frame):
            img = cv2.imread(frame_file[i])
            if img is None:
                raise ValueError(f"Failed to load image: {frame_file[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))  # From (H, W, C) to (C, H, W)
            frames.append(img)
        frames = np.concatenate(frames, axis=0)  # Shape: (num_frame * 3, H, W)

        # Generate corresponding heatmaps
        heatmaps = self._generate_heatmaps(coors, vis, h, w)  # Shape: (num_frame, H, W)

        # Normalize frames
        frames = frames.astype(np.float32) / 255.0

        logger.debug(f"Sample frame data for idx {idx}: shape={frames.shape}")
        logger.debug(f"Sample heatmap data for idx {idx}: shape={heatmaps.shape}")

        # Apply augmentations if any
        if self.augmentations:
            frames, heatmaps, coors = self.augmentations(frames, heatmaps, coors)

        return idx, frames, heatmaps, coors

    def _generate_heatmaps(self, coors, vis, h, w):
        """
        Generate heatmaps based on coordinates and visibility.

        Args:
            coors (np.ndarray): Coordinates, shape (num_frame, 2).
            vis (np.ndarray): Visibility flags, shape (num_frame).
            h (int): Image height.
            w (int): Image width.

        Returns:
            heatmaps (np.ndarray): Generated heatmaps, shape (num_frame, h, w).
        """
        heatmaps = np.zeros((self.num_frame, h, w), dtype=np.float32)

        for i in range(self.num_frame):
            if vis[i]:  # Only generate heatmap if the object is visible
                heatmap = self._get_heatmap(int(coors[i][0]), int(coors[i][1]), h, w)
                heatmaps[i] = heatmap[0]  # Extract heatmap from shape (1, h, w)

        return heatmaps

    def _get_heatmap(self, cx, cy, h, w):
        """
        Generate a single heatmap for given coordinates.

        Args:
            cx (int): X coordinate.
            cy (int): Y coordinate.
            h (int): Image height.
            w (int): Image width.

        Returns:
            heatmap (np.ndarray): Heatmap of shape (1, h, w).
        """
        x_grid, y_grid = np.meshgrid(
            np.linspace(1, w, w),
            np.linspace(1, h, h),
        )
        heatmap = (y_grid - (cy + 1)) ** 2 + (x_grid - (cx + 1)) ** 2
        heatmap[heatmap <= self.sigma**2] = self.mag
        heatmap[heatmap > self.sigma**2] = 0.0
        return heatmap.reshape(1, h, w)

import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
from typing import Any
from dataclasses import dataclass
from omegaconf import DictConfig

from dataset import TrackNetX_Dataset
from data_transforms import RandomCrop, RandomHorizontalFlip, Resize


def collate_fn(batch):
    idx_list = []
    frames_list = []
    heatmaps_list = []
    coors_list = []

    for idx, frames, heatmaps, coors in batch:
        idx_list.append(idx)
        frames_list.append(torch.from_numpy(frames.copy()))
        heatmaps_list.append(torch.from_numpy(heatmaps.copy()))
        coors_list.append(torch.from_numpy(coors.copy()))

    idxs = torch.tensor(idx_list, dtype=torch.long)
    frames_batch = torch.stack(frames_list, dim=0)
    heatmaps_batch = torch.stack(heatmaps_list, dim=0)
    coors_batch = torch.stack(coors_list, dim=0)

    return idxs, frames_batch, heatmaps_batch, coors_batch


@dataclass
class AugmentationConfig:
    """Configuration for data augmentations."""

    enabled: bool = True
    transforms: dict[str, Any] = None


class TrackNetXDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        num_frame: int = 3,
        stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        augmentation_config: DictConfig | None = None,
        mag: float = 1,
        sigma: float = 2.5,
        splits: tuple[str, ...] = ("train", "val", "test"),
    ):
        """
        PyTorch Lightning DataModule for TrackNetX dataset.

        Args:
            root_dir: Path to the dataset directory containing train/val/test splits
            num_frame: Number of frames in each sequence
            stride: Step size for the sliding window
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading
            augmentation_config: Configuration for data augmentations
            mag: Magnification factor for heatmap generation
            sigma: Sigma value for heatmap generation
            splits: Dataset splits to prepare
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_frame = num_frame
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_config = augmentation_config
        self.mag = mag
        self.sigma = sigma
        self.splits = splits
        self.datasets = {}

    def _build_transforms(self, split: str) -> list:
        """Build list of transforms based on configuration."""
        if not self.aug_config or not self.aug_config.enabled:
            return []

        transforms = []
        cfg = self.aug_config.transforms

        # Add Resize transform if enabled
        if cfg.resize.enabled:
            if cfg.resize.height is not None and cfg.resize.width is not None:
                transforms.append(Resize(target_height=cfg.resize.height, target_width=cfg.resize.width))

        # Add training-specific augmentations
        if split == "train":
            # Add RandomCrop if enabled
            if cfg.random_crop.enabled:
                if cfg.random_crop.height is not None and cfg.random_crop.width is not None:
                    transforms.append(
                        RandomCrop(
                            crop_height=cfg.random_crop.height,
                            crop_width=cfg.random_crop.width,
                            include_object_prob=cfg.random_crop.include_object_prob,
                        )
                    )

            # Add RandomHorizontalFlip if enabled
            if cfg.horizontal_flip.enabled:
                transforms.append(RandomHorizontalFlip(flip_prob=cfg.horizontal_flip.prob))
        return transforms

    def prepare_data(self):
        """Verify dataset files exist."""
        for split in self.splits:
            split_dir = self.root_dir / split
            required_files = [
                split_dir / "sequences.npy",
                split_dir / "coordinates.npy",
                split_dir / "visibility.npy",
            ]
            for file_path in required_files:
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing required file for {split} split: {file_path}")

    def setup(self, stage=None):
        """Load datasets for training, validation, and testing."""
        for split in self.splits:
            # Load the data files
            split_dir = self.root_dir / split
            frame_files = np.load(split_dir / "sequences.npy")
            coordinates = np.load(split_dir / "coordinates.npy")
            visibility = np.load(split_dir / "visibility.npy")

            logger.info(f"Loaded {split} split - {len(frame_files)} sequences")

            # Build transforms for this split
            transforms = self._build_transforms(split)

            # Create augmentation pipeline if there are any transforms
            augmentations = None
            if transforms:

                class ComposeTransforms:
                    def __init__(self, transforms):
                        self.transforms = transforms

                    def __call__(self, frames, heatmaps, coors):
                        for transform in self.transforms:
                            frames, heatmaps, coors = transform(frames, heatmaps, coors)
                        return frames, heatmaps, coors

                augmentations = ComposeTransforms(transforms)

            # Create dataset
            dataset = TrackNetX_Dataset(
                frame_files=frame_files,
                coordinates=coordinates,
                visibility=visibility,
                num_frame=self.num_frame,
                mag=self.mag,
                sigma=self.sigma,
                augmentations=augmentations,
            )
            self.datasets[split] = dataset

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if "val" in self.datasets:
            return DataLoader(
                self.datasets["val"],
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=collate_fn,
            )
        raise ValueError("Validation dataset is not available")

    def test_dataloader(self):
        """Return test dataloader."""
        if "test" in self.datasets:
            return DataLoader(
                self.datasets["test"],
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=collate_fn,
            )
        raise ValueError("Test dataset is not available")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from loguru import logger
    import matplotlib.pyplot as plt

    # Load config
    config_path = Path("configs/default_config.yaml")
    cfg = OmegaConf.load(config_path)

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Initialize DataModule
    data_module = TrackNetXDataModule(
        root_dir=cfg.ball_tracking.data.data_dir,
        num_frame=cfg.ball_tracking.data.num_frame,
        batch_size=cfg.ball_tracking.data.batch_size,
        num_workers=cfg.ball_tracking.data.num_workers,
        augmentation_config=cfg.ball_tracking.data.augmentation,
        mag=cfg.ball_tracking.data.mag,
        sigma=cfg.ball_tracking.data.sigma,
        splits=cfg.ball_tracking.data.splits,
    )

    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup()

    # Iterate train, val, and, test dataloaders
    for split in data_module.splits:
        if split == "train":
            dataloader = data_module.train_dataloader()
        elif split == "val":
            dataloader = data_module.val_dataloader()
        else:
            dataloader = data_module.test_dataloader()
        count = 0
        for _, frames, heatmaps, coords in tqdm(dataloader):
            count += 1
        logger.info(f"Number of batches in {split} split: {count}")
        logger.info(f"Frames shape: {frames.shape}")
        logger.info(f"Heatmaps shape: {heatmaps.shape}")
        logger.info(f"Coordinates shape: {coords.shape}")

    # For each frame in the sequence
    _, frames, heatmaps, coords = next(iter(data_module.train_dataloader()))
    frames = frames.numpy()[0]
    heatmaps = heatmaps.numpy()[0]
    coords = coords.numpy()[0]
    for i in range(cfg.ball_tracking.data.num_frame):  # 3 frames per sequence
        plt.subplot(cfg.ball_tracking.data.num_frame, 1, i + 1)

        # Get the RGB frame
        frame = frames[i * 3 : (i + 1) * 3].transpose(1, 2, 0)  # Convert to HWC

        # Create overlay
        plt.imshow(frame)

        # Overlay heatmap with alpha blending
        heatmap_overlay = plt.imshow(heatmaps[i], cmap="hot", alpha=0.5)

        # Add colorbar with fixed range from 0 to 1
        plt.colorbar(heatmap_overlay, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0])

        # Mark the actual ball position if visible
        if coords[i][0] != 0 and coords[i][1] != 0:  # If ball is visible
            plt.plot(coords[i][0], coords[i][1], "ro", mfc="none", mec="red", markersize=12, label="Ball Position")

        plt.title(f"Sample, Frame {i}")
        plt.axis("on")  # Show axes to help with coordinate interpretation

    # Save each sample to a separate file
    plt.savefig("dataset_visualization_dm.jpg", bbox_inches="tight", dpi=300)
    logger.info(f"Saved to: {Path.cwd() / 'dataset_visualization_dm.jpg'}")
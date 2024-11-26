import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tracknetx.dataset import TrackNetX_Dataset
from tracknetx.utils import list_dirs
import parse
import pandas as pd
from tqdm import tqdm
from loguru import logger
from tracknetx.data_transforms import RandomCrop, RandomHorizontalFlip, Resize


class TrackNetXDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        num_frame=3,
        stride=1,
        batch_size=32,
        num_workers=4,
        height=333,
        width=1000,
        mag=1,
        sigma=2.5,
        splits=("train", "test"),
        crop_height=333,  # Default crop height
        crop_width=1000,  # Default crop width
        include_object_prob=0.5,  # Probability to include the object in the crop
        flip_prob=0.5,  # Probability to apply flipping
    ):
        """
        PyTorch Lightning DataModule for TrackNetX dataset.

        Args:
            root_dir (str): Path to the dataset directory.
            num_frame (int): Number of frames in each sample.
            stride (int): Step size for the sliding window.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for data loading.
            height (int): Target image height after resizing.
            width (int): Target image width after resizing.
            mag (float): Magnification factor for heatmap generation.
            sigma (float): Sigma value for heatmap generation.
            splits (tuple): Dataset splits to prepare (default is ('train', 'test')).
            crop_height (int): Height of the random crop.
            crop_width (int): Width of the random crop.
            include_object_prob (float): Probability to include the tracked object in the crop.
            flip_prob (float): Probability of applying the horizontal flip.
        """
        super().__init__()
        self.root_dir = root_dir
        self.num_frame = num_frame
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.height = height
        self.width = width
        self.mag = mag
        self.sigma = sigma

        self.splits = splits

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.include_object_prob = include_object_prob
        self.flip_prob = flip_prob

        self.datasets = {}

    def prepare_data(self):
        """
        Prepare data if necessary.

        This method is called only from a single process to avoid race conditions.
        """
        pass  # No caching, so nothing is required here

    def setup(self, stage=None):
        """
        Load datasets for training, validation, and testing.

        This method is called on every GPU.
        """
        for split in self.splits:
            frame_files, coordinates, visibility = self._gen_frame_files(split)

            # Define augmentations
            augmentations_list = []

            # Add Resize transform
            resize_transform = Resize(target_height=self.height, target_width=self.width)
            augmentations_list.append(resize_transform)

            if split == "train":
                # Add training augmentations
                augmentations_list.extend(
                    [
                        RandomCrop(
                            crop_height=self.crop_height,
                            crop_width=self.crop_width,
                            include_object_prob=self.include_object_prob,
                        ),
                        RandomHorizontalFlip(flip_prob=self.flip_prob),
                    ]
                )

            # Combine augmentations
            class ComposeTransforms:
                def __init__(self, transforms):
                    self.transforms = transforms

                def __call__(self, frames, heatmaps, coors):
                    for transform in self.transforms:
                        frames, heatmaps, coors = transform(frames, heatmaps, coors)
                    return frames, heatmaps, coors

            augmentations = ComposeTransforms(augmentations_list)

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
        return DataLoader(
            self.datasets.get("train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if "val" in self.datasets:
            return DataLoader(
                self.datasets["val"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        elif "test" in self.datasets:
            # If no validation dataset, optionally use test dataset
            return self.test_dataloader()
        else:
            raise ValueError("Validation dataset is not available.")

    def test_dataloader(self):
        if "test" in self.datasets:
            return DataLoader(
                self.datasets["test"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        else:
            raise ValueError("Test dataset is not available.")

    def _gen_frame_files(self, split):
        """
        Generate frame file paths, coordinates, and visibility information.

        Args:
            split (str): The dataset split being processed.

        Returns:
            Tuple[np.ndarray]: frame_files, coordinates, visibility arrays.
        """
        rally_dirs = self._get_rally_dirs(split)
        frame_files = []
        coordinates = []
        visibility = []

        max_frame = 249  # Maximum number of frames (from 0.png to 249.png)

        for rally_dir in tqdm(rally_dirs, desc=f"Generating {split} data"):
            parsed = parse.parse("{}/frame/{}", rally_dir)
            if parsed:
                match_dir, rally_id = parsed
            else:
                logger.info(f"Failed to parse rally directory: {rally_dir}")
                continue
            csv_file = os.path.join(match_dir, "csv", f"{rally_id}_ball.csv")
            try:
                label_df = pd.read_csv(csv_file, encoding="utf8")
                label_df = label_df.sort_values(by="Frame").fillna(0)
            except FileNotFoundError:
                logger.info(
                    f"Label file {rally_id}_ball.csv not found. Skipping {rally_dir}"
                )
                continue

            # Limit frames to max_frame
            label_df = label_df[label_df["Frame"] <= max_frame].copy()

            frame_paths = [
                os.path.join(rally_dir, f"{int(f_id)}.png")
                for f_id in label_df["Frame"]
            ]
            x_coords = label_df["X"].values
            y_coords = label_df["Y"].values
            visibilities = label_df["Visibility"].values

            if not (
                len(frame_paths) == len(x_coords) == len(y_coords) == len(visibilities)
            ):
                logger.info(f"Data length mismatch in {csv_file}. Skipping.")
                continue

            # Generate sequences using sampling interval defined by stride
            max_i = len(frame_paths) - (self.num_frame - 1) * self.stride
            for i in range(0, max_i):
                # Sample frames at intervals of 'stride'
                tmp_frames = [
                    frame_paths[i + j * self.stride] for j in range(self.num_frame)
                ]
                tmp_coors = [
                    (x_coords[i + j * self.stride], y_coords[i + j * self.stride])
                    for j in range(self.num_frame)
                ]
                tmp_vis = [
                    visibilities[i + j * self.stride] for j in range(self.num_frame)
                ]

                # Ensure all frames exist
                if all(os.path.exists(fp) for fp in tmp_frames):
                    frame_files.append(tmp_frames)
                    coordinates.append(tmp_coors)
                    visibility.append(tmp_vis)
                else:
                    missing_frames = [fp for fp in tmp_frames if not os.path.exists(fp)]
                    logger.info(f"Missing frames: {missing_frames}. Skipping sequence.")

        if len(frame_files) == 0:
            logger.info(f"No data found for split '{split}'.")
            return None, None, None

        # Convert lists to numpy arrays
        frame_files = np.array(frame_files)  # Shape: (N, num_frame)
        coordinates = np.array(
            coordinates, dtype=np.float32
        )  # Shape: (N, num_frame, 2)
        visibility = np.array(visibility, dtype=np.float32)  # Shape: (N, num_frame)

        return frame_files, coordinates, visibility

    def _get_rally_dirs(self, split):
        """
        Retrieve a list of rally directories for a given dataset split.

        Args:
            split (str): The dataset split ('train', 'val', 'test').

        Returns:
            List[str]: List of paths to rally directories.
        """
        split_dir = os.path.join(self.root_dir, split)
        match_dirs = list_dirs(split_dir)
        match_dirs = sorted(match_dirs, key=lambda s: int(s.split("match")[-1]))
        rally_dirs = []
        for match_dir in match_dirs:
            rally_subdirs = list_dirs(os.path.join(match_dir, "frame"))
            rally_dirs.extend(rally_subdirs)
        return rally_dirs

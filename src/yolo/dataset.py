"""Module for converting MOT format data to YOLO format."""

import cv2
import pandas as pd
import yaml
import shutil
import random
from loguru import logger
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, frame_interval: int = 1) -> None:
    """Extract frames from video at specified intervals.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every nth frame
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    for _ in tqdm(range(total_frames), desc="Extracting frames", total=total_frames // frame_interval):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        frame_count += 1

    cap.release()
    logger.info(f"Extracted {frame_count} frames from {video_path}")


def split_dataset(dataset_dir: Path, train_ratio: float = 0.8) -> None:
    """Split dataset into train and validation sets.

    Args:
        dataset_dir: Path to dataset directory containing images and labels
        train_ratio: Ratio of images to use for training
    """
    # Create train and val directories
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            (dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list((dataset_dir / "images").glob("*.jpg"))
    random.shuffle(image_files)

    # Split files
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    logger.info(f"Splitting dataset: {len(train_files)} train, {len(val_files)} validation images")

    # Move files to respective directories
    train_labels = 0
    val_labels = 0

    for files, split in [(train_files, "train"), (val_files, "val")]:
        for img_path in files:
            # Move image
            shutil.move(str(img_path), str(dataset_dir / split / "images" / img_path.name))

            # Move corresponding label
            label_path = dataset_dir / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                # Count annotations in label file
                with open(label_path, "r") as f:
                    num_annotations = len(f.readlines())

                if split == "train":
                    train_labels += num_annotations
                else:
                    val_labels += num_annotations

                shutil.move(str(label_path), str(dataset_dir / split / "labels" / label_path.name))

    # Remove original directories
    shutil.rmtree(dataset_dir / "images")
    shutil.rmtree(dataset_dir / "labels")

    # Update dataset.yaml
    dataset_yaml = dataset_dir / "dataset.yaml"
    with open(dataset_yaml, "r") as f:
        lines = f.readlines()

    with open(dataset_yaml, "w") as f:
        for line in lines:
            if line.startswith("train:"):
                f.write("train: train/images\n")
            elif line.startswith("val:"):
                f.write("val: val/images\n")
            else:
                f.write(line)

    # Log detailed dataset statistics
    logger.info("Dataset split complete. Statistics:")
    logger.info(f"Training set:")
    logger.info(f"  - {len(train_files)} images")
    logger.info(f"  - {train_labels} total player annotations")
    logger.info(f"  - {train_labels / len(train_files):.1f} players per image (avg)")
    logger.info(f"Validation set:")
    logger.info(f"  - {len(val_files)} images")
    logger.info(f"  - {val_labels} total player annotations")
    logger.info(f"  - {val_labels / len(val_files):.1f} players per image (avg)")
    logger.info(f"Total: {len(train_files) + len(val_files)} images, {train_labels + val_labels} annotations")


def create_yolo_dataset(
    match_id: str,
    half: str,
    frame_interval: int = 1,
    train_ratio: float = 0.8,
    base_dir: Path | str | None = None,
) -> None:
    """Convert MOT format data to YOLO format for training.

    Args:
        match_id: Match ID
        half: Which half of the match (1st or 2nd)
        frame_interval: Extract every nth frame
        train_ratio: Ratio of images to use for training (default: 0.8)
        base_dir: Base directory for data (defaults to data/interim/{match_id})
    """
    # Setup paths
    base_dir = Path(base_dir) if base_dir else Path(f"data/interim/{match_id}")
    video_path = base_dir / f"{match_id}_panorama_{half}_half.mp4"
    mot_file = base_dir / f"{match_id}_ground_truth_mot_{half}_half_distorted.csv"
    output_dir = base_dir / f"ultralytics_format_{half}_half_distorted"

    # Create directory structure
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    logger.info(f"Extracting frames from {video_path}")
    extract_frames(video_path, images_dir, frame_interval)

    # Read MOT format data
    # Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z,class_name
    logger.info(f"Converting annotations from {mot_file}")
    df = pd.read_csv(mot_file, header=None)
    df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z", "class_name"]

    # Log MOT file statistics
    total_frames_mot = len(df["frame"].unique())
    total_annotations = len(df)
    logger.info(f"MOT file statistics:")
    logger.info(f"  - Total frames: {total_frames_mot}")
    logger.info(f"  - Total annotations: {total_annotations}")
    logger.info(f"  - Frame range: {df['frame'].min()} to {df['frame'].max()}")
    logger.info(f"  - Average annotations per frame: {total_annotations / total_frames_mot:.1f}")

    # Get video dimensions for normalization
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.info(f"Video statistics:")
    logger.info(f"  - Total frames: {total_frames_video}")
    logger.info(f"  - Frame interval: {frame_interval}")
    logger.info(f"  - Expected frames after interval: {total_frames_video // frame_interval}")
    logger.info(f"  - Resolution: {width}x{height}")

    # Process each frame
    unique_frames = sorted(df["frame"].unique())
    frames_with_annotations = 0
    total_boxes = 0
    skipped_frames = 0

    for frame_idx in tqdm(unique_frames, desc="Converting annotations"):
        if frame_idx % frame_interval != 0:
            skipped_frames += 1
            continue

        frame_data = df[df["frame"] == frame_idx]

        # Verify corresponding image exists
        image_path = images_dir / f"frame_{frame_idx:06d}.jpg"
        if not image_path.exists():
            logger.warning(f"Missing image for frame {frame_idx}")
            continue

        frames_with_annotations += 1
        total_boxes += len(frame_data)

        # Create YOLO format label file
        label_path = labels_dir / f"frame_{frame_idx:06d}.txt"
        with open(label_path, "w") as f:
            for _, row in frame_data.iterrows():
                # Convert bbox to YOLO format (normalized xcenter, ycenter, width, height)
                x_center = (row["bb_left"] + row["bb_width"] / 2) / width
                y_center = (row["bb_top"] + row["bb_height"] / 2) / height
                w = row["bb_width"] / width
                h = row["bb_height"] / height

                # Class is always 0 (player) for now
                class_idx = 0

                # Write YOLO format line: class x_center y_center width height
                f.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")

    logger.info(f"Annotation conversion statistics:")
    logger.info(f"  - Frames with annotations: {frames_with_annotations}")
    logger.info(f"  - Skipped frames (due to interval): {skipped_frames}")
    logger.info(f"  - Total bounding boxes: {total_boxes}")
    logger.info(f"  - Average boxes per frame: {total_boxes / frames_with_annotations:.1f}")

    # Create dataset YAML file
    dataset_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images",  # Will be split later
        "val": "images",  # Will be split later
        "names": {
            0: "player"  # Only one class for now
        },
    }

    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    # Split dataset into train/val
    logger.info("Splitting dataset into train and validation sets...")
    split_dataset(output_dir, train_ratio)

    logger.success(f"Dataset created and split at {output_dir}")

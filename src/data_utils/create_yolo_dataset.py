"""Script for preparing datasets for YOLOv8 training from videos and MOT format files."""

import shutil
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from rich.progress import Progress
from sklearn.model_selection import train_test_split

from ..video_utils import get_fps, get_total_frames


def extract_frames_generator(
    video_path: Path,
    frame_interval: int = 1,
    target_size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Generate frames from a video file using DeFFcode's FFdecoder.

    Args:
        video_path (Path): Path to the video file.
        frame_interval (int): Extract every Nth frame.
        target_size (tuple[int, int] | None): Optional target size for frames (width, height).
                                             If None, original frame size is preserved.

    Yields:
        Iterator[tuple[int, np.ndarray]]: Frame number and frame data.

    Raises:
        RuntimeError: If video cannot be opened or FFmpeg is not found.
    """
    from deffcode import FFdecoder

    # Configure FFdecoder parameters
    ffparams = {
        "-vsync": "0",  # Disable video sync to maintain frame accuracy
        "-vf": f"select=not(mod(n\\,{frame_interval}))",  # Select every Nth frame
    }

    if target_size is not None:
        width, height = target_size
        # Add scale filter after frame selection
        ffparams["-vf"] = f"{ffparams['-vf']},scale={width}:{height}"

    try:
        # Initialize decoder with BGR24 format (compatible with OpenCV)
        decoder = FFdecoder(str(video_path), frame_format="bgr24", **ffparams).formulate()

        frame_count = 0
        for frame in decoder.generateFrame():
            if frame is None:
                logger.debug("No more frames to read.")
                break

            # Since we're already selecting frames with FFmpeg,
            # frame_count directly corresponds to the original frame number
            frame_number = frame_count * frame_interval
            yield frame_number, frame
            frame_count += 1

    except Exception as e:
        logger.error(f"Failed to process video {video_path}: {str(e)}")
        raise
    finally:
        if "decoder" in locals():
            decoder.terminate()
            logger.info(f"Released video capture for {video_path}")


def get_class_id(track_id: int) -> int:
    """
    Map MOT track IDs to YOLO class IDs.

    Args:
        track_id: The track ID from MOT format

    Returns:
        int: Class ID (0: team1, 1: team2, 2: ball)
    """
    if 1 <= track_id <= 22:
        return 0  # Team 1
    elif track_id == 23:
        return 1  # Ball
    else:
        logger.warning(f"Unknown track ID: {track_id}, defaulting to -1")
        return -1


def process_video_and_mot(
    video_path: Path,
    mot_path: Path,
    output_dir: Path,
    frame_interval: int = 1,
    target_size: tuple[int, int] | None = None,
) -> list[Path]:
    """
    Process a video file and its MOT annotations to create YOLO format dataset.

    Args:
        video_path: Path to the video file.
        mot_path: Path to the MOT format annotation file.
        output_dir: Directory to save extracted frames and labels.
        frame_interval: Extract every Nth frame.
        target_size: Optional target size for frames (width, height).
                    If None, original frame size is preserved.

    Returns:
        list[Path]: List of paths to extracted frames.

    Raises:
        FileNotFoundError: If video or MOT file doesn't exist.
        ValueError: If MOT file format is invalid.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not mot_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Read MOT data
    mot_data = pd.read_csv(
        mot_path,
        names=["frame_id", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
    )

    # Get video info
    total_frames = get_total_frames(video_path)
    fps = get_fps(video_path)
    total_seconds = total_frames / fps
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    logger.info(
        f"Processing video: {video_path.name} ({total_frames} frames, {fps} FPS), {minutes:.0f} minutes {seconds:.0f} seconds"
    )

    frame_paths = []

    with Progress() as progress:
        task = progress.add_task(f"Processing {video_path.name}...", total=total_frames / frame_interval)

        # Process frames
        for frame_id, frame in extract_frames_generator(
            video_path, frame_interval=frame_interval, target_size=target_size
        ):
            # Save frame
            frame_path = image_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)

            # Get annotations for this frame
            frame_dets = mot_data[mot_data["frame_id"] == frame_id]

            if not frame_dets.empty:
                # Create YOLO format labels
                img_height, img_width = frame.shape[:2]
                label_path = label_dir / f"frame_{frame_id:06d}.txt"

                with open(label_path, "w") as f:
                    for _, det in frame_dets.iterrows():
                        # Convert bbox to YOLO format (normalized coordinates)
                        x_center = (det["bb_left"] + det["bb_width"] / 2) / img_width
                        y_center = (det["bb_top"] + det["bb_height"] / 2) / img_height
                        width = det["bb_width"] / img_width
                        height = det["bb_height"] / img_height

                        # Ensure coordinates are within bounds
                        x_center = np.clip(x_center, 0, 1)
                        y_center = np.clip(y_center, 0, 1)
                        width = np.clip(width, 0, 1)
                        height = np.clip(height, 0, 1)

                        # Get class ID based on track ID
                        class_id = get_class_id(det["track_id"])
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

            progress.advance(task)

    return frame_paths


def create_yolo_dataset(
    video_path: str,
    mot_path: str,
    output_dir: str,
    frame_interval: int = 5,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    frame_width: int | None = None,
    frame_height: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a YOLO format dataset from a video and MOT format file."""
    if not abs(train_split + val_split + test_split - 1.0) < 1e-9:
        raise ValueError("Train, validation, and test splits must sum to 1.0")

    video_path = Path(video_path)
    mot_path = Path(mot_path)
    output_dir = Path(output_dir)

    # Clean up temp directory
    if overwrite:
        shutil.rmtree(output_dir / "temp", ignore_errors=True)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    temp_dir = output_dir / "temp"

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # Create target size tuple only if both dimensions are provided
    target_size = None
    if frame_width is not None and frame_height is not None:
        target_size = (frame_width, frame_height)

    frame_paths = process_video_and_mot(
        video_path=video_path,
        mot_path=mot_path,
        output_dir=temp_dir,
        frame_interval=frame_interval,
        target_size=target_size,
    )

    # Split data
    train_val_files, test_files = train_test_split(frame_paths, test_size=test_split, random_state=42)

    train_files, val_files = train_test_split(
        train_val_files, test_size=val_split / (train_split + val_split), random_state=42
    )

    # Organize files into splits
    splits = {"train": train_files, "val": val_files, "test": test_files}

    with Progress() as progress:
        for split_name, files in splits.items():
            task = progress.add_task(f"Organizing {split_name} split...", total=len(files))

            for frame_path in files:
                # Get corresponding label path
                label_path = temp_dir / "labels" / f"{frame_path.stem}.txt"

                # Move to final location
                dest_img = images_dir / split_name / frame_path.name
                dest_label = labels_dir / split_name / f"{frame_path.stem}.txt"

                shutil.move(frame_path, dest_img)
                if label_path.exists():
                    shutil.move(label_path, dest_label)

                progress.advance(task)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": str(Path("images/train").absolute()),
        "val": str(Path("images/val").absolute()),
        "test": str(Path("images/test").absolute()),
        "names": {0: "person", 1: "ball"},
        "nc": 2,  # number of classes
    }

    with open(output_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info(f"Dataset created successfully at {output_dir}")
    logger.info(f"Use {output_dir}/data.yaml for training")

    return output_dir

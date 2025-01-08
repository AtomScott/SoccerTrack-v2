import os
import cv2
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
from loguru import logger
import shutil
import sys
import moviepy.editor as mpe


import os
from pathlib import Path
import logging
from tqdm import tqdm
import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_stride: int = 1,
    sequence_stride: int = 1,
    sequence_length: int = 3,
    max_num_sequences: int = 1000000,
    downscale_factor: int = 2,
    use_seek: bool = True,
) -> dict[int, list[int]]:
    """
    Extract frames from a video file for TrackNetX dataset creation.

    You can choose between two methods:
    - Sequential: Reads frames one by one (original method).
    - Seek-based: Calculates the required frames and seeks directly to them (faster, but depends on codec).

    Args:
        video_path: Path to the input video file
        output_dir: Directory where extracted frames will be saved
        frame_stride: Number of frames to skip between frames in a sequence
        sequence_stride: Number of frames to skip between sequences
        sequence_length: Number of consecutive frames in each sequence
        max_num_sequences: Maximum number of sequences to extract
        downscale_factor: Factor to downscale the frames
        use_seek: If True, use direct frame seeking. If False, read sequentially.

    Returns:
        Dictionary mapping sequence IDs to lists of frame indices in that sequence
    """
    # Original sequential method using OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate frames needed for one sequence
    frames_per_sequence = (sequence_length - 1) * frame_stride + 1
    if not use_seek:
        frame_idx = 0
        extracted_sequences = 0
        extracted_frames = 0
        sequence_info = {}
        current_sequence_frames = []

        with tqdm(total=max_num_sequences, desc=f"Extracting frames from {video_path.name}") as pbar:
            while frame_idx < total_frames and extracted_sequences < max_num_sequences:
                ret, frame = cap.read()
                if not ret:
                    break

                pos_in_sequence = frame_idx % sequence_stride

                # Check if this frame belongs to the current sequence and is at a stride position
                if pos_in_sequence < frames_per_sequence and pos_in_sequence % frame_stride == 0:
                    frame_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                    frame = cv2.resize(frame, (frame.shape[1] // downscale_factor, frame.shape[0] // downscale_factor))
                    cv2.imwrite(frame_filename, frame)
                    extracted_frames += 1
                    current_sequence_frames.append(frame_idx)

                    # If we've reached the last frame of this sequence
                    if pos_in_sequence == frames_per_sequence - 1:
                        sequence_info[extracted_sequences] = current_sequence_frames
                        current_sequence_frames = []
                        extracted_sequences += 1
                        pbar.update(1)

                frame_idx += 1

        cap.release()
        logger.info(
            f"Extracted {extracted_frames} frames ({extracted_sequences} sequences) "
            f"with frame_stride {frame_stride} and sequence_stride {sequence_stride} "
            f"for sequences of length {sequence_length} using {'seek' if use_seek else 'sequential'} mode."
        )
        return sequence_info

    else:
        # MoviePy-based seeking method
        video = mpe.VideoFileClip(str(video_path))
        total_frames = int(video.duration * video.fps)
        fps = video.fps

        # Pre-calculate frame indices for sequences
        all_sequences_indices = []
        seq_id = 0

        while True:
            start_frame = seq_id * sequence_stride
            last_required_frame = start_frame + (frames_per_sequence - 1) * frame_stride
            if start_frame >= total_frames or last_required_frame >= total_frames:
                break
            if seq_id >= max_num_sequences:
                break

            seq_indices = [start_frame + i * frame_stride for i in range(sequence_length)]
            all_sequences_indices.append((seq_id, seq_indices))
            seq_id += 1

        sequence_info = {}
        extracted_frames = 0

        with tqdm(total=len(all_sequences_indices), desc=f"Extracting frames from {video_path.name}") as pbar:
            for seq_id, seq_indices in all_sequences_indices:
                current_sequence_frames = []
                for frame_idx in seq_indices:
                    # Convert frame index to timestamp
                    timestamp = frame_idx / fps

                    try:
                        # Get frame at timestamp
                        frame = video.get_frame(timestamp)

                        # Convert from RGB to BGR for OpenCV compatibility
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Resize frame
                        frame = cv2.resize(
                            frame, (frame.shape[1] // downscale_factor, frame.shape[0] // downscale_factor)
                        )

                        # Save frame
                        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        current_sequence_frames.append(frame_idx)
                        extracted_frames += 1

                    except Exception as e:
                        logger.error(f"Failed to extract frame {frame_idx}: {e}")
                        break

                if len(current_sequence_frames) == sequence_length:
                    sequence_info[seq_id] = current_sequence_frames

                pbar.update(1)

        # Clean up
        video.close()

        logger.info(
            f"Extracted {extracted_frames} frames ({len(sequence_info)} sequences) "
            f"with frame_stride {frame_stride} and sequence_stride {sequence_stride} "
            f"for sequences of length {sequence_length} using {'seek' if use_seek else 'sequential'} mode."
        )
        return sequence_info


def parse_mot_annotations(mot_path: Path) -> pd.DataFrame:
    """Parse Multiple Object Tracking (MOT) format annotations.

    Args:
        mot_path: Path to the MOT annotation file

    Returns:
        DataFrame containing parsed MOT annotations with columns:
        - frame_id: Frame number
        - object_id: Unique object identifier
        - bb_left, bb_top: Bounding box top-left coordinates
        - bb_width, bb_height: Bounding box dimensions
        - confidence: Detection confidence
        - x, y, z: 3D coordinates (if available)
    """
    column_names = ["frame_id", "object_id", "bb_left", "bb_top", "bb_width", "bb_height", "confidence", "x", "y", "z"]
    mot_df = pd.read_csv(mot_path, header=None, names=column_names)

    # filter object_id == 23
    mot_df = mot_df[mot_df["object_id"] == 23]
    return mot_df


def generate_sequences(
    frame_dir: Path,
    mot_df: pd.DataFrame,
    sequence_info: dict[int, list[int]],
    sequence_length: int = 3,
    downscale_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate frame sequences and corresponding ball coordinates for TrackNetX training.

    Args:
        frame_dir: Directory containing extracted video frames
        mot_df: DataFrame with MOT format annotations
        sequence_info: Dictionary mapping sequence IDs to lists of frame indices
        sequence_length: Number of frames in each sequence
        downscale_factor: Factor to downscale the frames
    """
    sequences = []
    coordinates = []
    visibility = []

    logger.info(f"Generating sequences from {len(sequence_info)} extracted sequences")

    # Process each sequence using the stored frame indices
    for seq_id, frame_indices in sequence_info.items():
        if len(frame_indices) != sequence_length:
            logger.warning(f"Sequence {seq_id} has {len(frame_indices)} frames, expected {sequence_length}")
            continue

        seq_frame_files = []
        seq_coords = []
        seq_vis = []
        valid_sequence = True

        # Process each frame in the sequence
        for frame_idx in frame_indices:
            frame_file = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")

            if not os.path.exists(frame_file):
                valid_sequence = False
                logger.warning(f"Missing frame file: {frame_file}")
                break

            seq_frame_files.append(frame_file)

            # Get corresponding MOT entry
            mot_entries = mot_df[mot_df["frame_id"] == frame_idx + 1]
            if not mot_entries.empty:
                mot_entry = mot_entries.iloc[0]
                x_center = (mot_entry["bb_left"] + mot_entry["bb_width"] / 2) / downscale_factor
                y_center = (mot_entry["bb_top"] + mot_entry["bb_height"] / 2) / downscale_factor
                seq_coords.append([x_center, y_center])
                seq_vis.append(1)
            else:
                seq_coords.append([0, 0])
                seq_vis.append(0)

        if valid_sequence:
            sequences.append(seq_frame_files)
            coordinates.append(seq_coords)
            visibility.append(seq_vis)

            if len(sequences) <= 3:  # Log first few sequences for debugging
                logger.info(f"Sequence {len(sequences)}: {[os.path.basename(f) for f in seq_frame_files]}")

    total_sequences = len(sequences)
    total_frames_in_sequences = total_sequences * sequence_length

    logger.info(f"Generated {total_sequences} sequences")
    logger.info(f"Total frames in sequences: {total_frames_in_sequences}")

    return np.array(sequences), np.array(coordinates), np.array(visibility)


if __name__ == "__main__":
    # Load configuration
    config = OmegaConf.load("configs/default_config.yaml").ball_tracking.data
    video_paths = [Path(video_path) for video_path in config.video_paths]
    mot_paths = [Path(mot_path) for mot_path in config.mot_paths]
    output_dir = Path(config.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        user_input = input(f"Output directory {output_dir} already exists. Do you want to overwrite it? (y/n): ")
        if user_input.lower() != "y":
            logger.info("Process stopped by user")
            sys.exit(0)
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # Create directories for each split
    split_dirs = {}
    for split in ["train", "val", "test"]:
        split_dirs[split] = output_dir / split
        os.makedirs(split_dirs[split], exist_ok=True)

    all_sequences = []
    all_coordinates = []
    all_visibility = []

    # Process each video and corresponding MOT file
    for video_file, mot_file in zip(video_paths, mot_paths):
        if not mot_file.exists():
            raise ValueError(f"Warning: No MOT file found for {video_file.name}")

        # Extract frames and get sequence information
        frame_dir = output_dir / "frames" / video_file.stem
        sequence_info = extract_frames(
            video_file,
            frame_dir,
            frame_stride=config.frame_stride,
            sequence_stride=config.sequence_stride,
            sequence_length=config.num_frame,
            max_num_sequences=config.max_num_sequences,
            downscale_factor=config.downscale_factor,
        )

        # Parse MOT annotations
        mot_df = parse_mot_annotations(mot_file)

        # Generate sequences using the sequence information
        sequences, coordinates, visibility = generate_sequences(
            frame_dir,
            mot_df,
            sequence_info,
            sequence_length=config.num_frame,
            downscale_factor=config.downscale_factor,
        )

        all_sequences.extend(sequences)
        all_coordinates.extend(coordinates)
        all_visibility.extend(visibility)
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_coordinates = np.array(all_coordinates)
    all_visibility = np.array(all_visibility)

    # Calculate split sizes
    splits = config.splits
    total_sequences = len(all_sequences)
    train_size = int(total_sequences * splits.train)
    val_size = int(total_sequences * splits.val)
    test_size = total_sequences - train_size - val_size

    logger.info(f"Total sequences: {total_sequences}")
    logger.info(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Generate random indices for splits
    indices = np.random.permutation(total_sequences)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Save splits
    splits = {
        "train": (train_indices, split_dirs["train"]),
        "val": (val_indices, split_dirs["val"]),
        "test": (test_indices, split_dirs["test"]),
    }

    for split_name, (indices, split_dir) in splits.items():
        # Save numpy arrays for this split
        np.save(split_dir / "sequences.npy", all_sequences[indices])
        np.save(split_dir / "coordinates.npy", all_coordinates[indices])
        np.save(split_dir / "visibility.npy", all_visibility[indices])

        # Copy frame files for this split
        frames_dir = split_dir / "frames"
        os.makedirs(frames_dir, exist_ok=True)

        # Get unique frame files used in this split
        split_sequences = all_sequences[indices]
        unique_frames = set()
        for seq in split_sequences:
            unique_frames.update(seq)

        # Copy frames
        for frame_path in unique_frames:
            dest_path = frames_dir / Path(frame_path).name
            if not dest_path.exists():  # Avoid copying if already exists
                shutil.copy2(frame_path, dest_path)

        logger.info(f"Created {split_name} split with {len(indices)} sequences")
        logger.info(f"Copied {len(unique_frames)} unique frames to {split_name} split")

    logger.info("Dataset creation completed successfully")

"""Functions for trimming videos into first and second halves using padding info.

Usage:
    There are two ways to run the video trimming:

    1. Using the Python module directly:
    ```bash
    python -m src.main command=trim_video_into_halves \\
        trim_video_into_halves.match_id=117093 \\
        trim_video_into_halves.input_video_path="data/raw/117093/117093_panorama.mp4" \\
        trim_video_into_halves.padding_info_path="data/raw/117093/117093_padding_info.csv" \\
        trim_video_into_halves.output_dir="data/interim/117093"
    ```

    2. Using the convenience shell script (recommended):
    ```bash
    # First ensure the script is executable
    chmod +x scripts/trim_video_into_halves.sh
    
    # Then run with match ID
    ./scripts/trim_video_into_halves.sh 117093
    ```

    The shell script automatically constructs the correct paths based on the match ID.
    Required file structure:
    - Input video: data/raw/<match_id>/<match_id>_panorama.mp4
    - Padding info: data/raw/<match_id>/<match_id>_padding_info.csv
    - Output directory: data/interim/<match_id>/
"""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger


def load_padding_info(padding_info_path: Path | str) -> List[Dict[str, str]]:
    """
    Load padding info from CSV file.

    Args:
        padding_info_path: Path to the padding info CSV file.

    Returns:
        List of dictionaries containing padding info for each half.
    """
    with open(padding_info_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def convert_time_to_seconds(time_ms: str) -> float:
    """
    Convert time from milliseconds string to seconds float.

    Args:
        time_ms: Time in milliseconds as string.

    Returns:
        Time in seconds as float.
    """
    return float(time_ms) / 1000.0


def trim_video_half(
    input_video_path: Path | str,
    output_video_path: Path | str,
    padding_ms: str,
    start_time_ms: str,
    end_time_ms: str,
) -> None:
    """
    Trim a video half using padding and match time information.

    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path where the trimmed video will be saved.
        padding_ms: Padding time in milliseconds.
        start_time_ms: Start match time in milliseconds.
        end_time_ms: End match time in milliseconds.

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails.
        ValueError: If end_time is less than or equal to start_time.
    """
    # Convert times to seconds
    padding_sec = convert_time_to_seconds(padding_ms)
    start_time_sec = convert_time_to_seconds(start_time_ms)
    end_time_sec = convert_time_to_seconds(end_time_ms)

    # Calculate actual start and end times
    actual_start = padding_sec
    actual_end = padding_sec + (end_time_sec - start_time_sec)

    if actual_end <= actual_start:
        raise ValueError("End time must be greater than start time")

    duration = actual_end - actual_start
    cmd = [
        "ffmpeg",
        "-i",
        str(input_video_path),
        "-ss",
        str(actual_start),
        "-t",
        str(duration),
        "-c",
        "copy",  # Use stream copy for faster processing
        str(output_video_path),
        "-y",  # Overwrite output file if it exists
    ]

    try:
        logger.info(f"Trimming video from {actual_start}s to {actual_end}s")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully saved trimmed video to {output_video_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to trim video: {e.stderr}")
        raise


def trim_video_into_halves(
    match_id: str,
    input_video_path: Path | str,
    padding_info_path: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """
    Trim a video into first and second halves using padding info.

    Args:
        match_id: Match ID for naming output files.
        input_video_path: Path to the input video file.
        padding_info_path: Path to the padding info CSV file.
        output_dir: Directory to save the trimmed videos.

    Returns:
        Tuple of (first_half_path, second_half_path).

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails.
        ValueError: If padding info is invalid.
        FileNotFoundError: If input files don't exist.
    """
    try:
        # Convert paths to Path objects
        input_video_path = Path(input_video_path)
        padding_info_path = Path(padding_info_path)
        output_dir = Path(output_dir)

        # Check input files exist
        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        if not padding_info_path.exists():
            raise FileNotFoundError(f"Padding info not found: {padding_info_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load padding info
        padding_info = load_padding_info(padding_info_path)
        if len(padding_info) != 2:
            raise ValueError("Padding info must contain exactly two rows (first and second half)")

        # Set up output paths
        first_half_path = output_dir / f"{match_id}_panorama_1st_half.mp4"
        second_half_path = output_dir / f"{match_id}_panorama_2nd_half.mp4"

        # Process first half
        first_half = next(info for info in padding_info if info["Event Period"] == "FIRST_HALF")
        trim_video_half(
            input_video_path=input_video_path,
            output_video_path=first_half_path,
            padding_ms=first_half["Padding"],
            start_time_ms=first_half["Start Match Time"],
            end_time_ms=first_half["End Match Time"],
        )

        # Process second half
        second_half = next(info for info in padding_info if info["Event Period"] == "SECOND_HALF")
        trim_video_half(
            input_video_path=input_video_path,
            output_video_path=second_half_path,
            padding_ms=second_half["Padding"],
            start_time_ms=second_half["Start Match Time"],
            end_time_ms=second_half["End Match Time"],
        )

        logger.info(f"Successfully created:\n  First half: {first_half_path}\n  Second half: {second_half_path}")
        return first_half_path, second_half_path

    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        raise


def trim_video_into_halves_command(
    match_id: str,
    input_video_path: Path | str,
    padding_info_path: Path | str,
    output_dir: Path | str,
) -> None:
    """
    Trim a video into first and second halves using padding info.

    Args:
        match_id: Match ID for naming output files.
        input_video_path: Path to the input video file.
        padding_info_path: Path to the padding info CSV file.
        output_dir: Directory to save the trimmed videos.

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails.
        ValueError: If padding info is invalid.
        FileNotFoundError: If input files don't exist.
    """
    try:
        first_half, second_half = trim_video_into_halves(
            match_id=match_id,
            input_video_path=input_video_path,
            padding_info_path=padding_info_path,
            output_dir=output_dir,
        )
        logger.info(f"Successfully created:\n  First half: {first_half}\n  Second half: {second_half}")
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        raise

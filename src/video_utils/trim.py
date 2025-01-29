"""Functions for trimming video files."""

import subprocess
from pathlib import Path

import typer
from loguru import logger


def trim_video(input_video_path: Path | str, output_video_path: Path | str, start_time: float, end_time: float) -> None:
    """
    Trims a video to a specified start and end time using ffmpeg.

    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path where the trimmed video will be saved.
        start_time: Start time in seconds to trim the video from.
        end_time: End time in seconds to trim the video to.

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails.
        ValueError: If end_time is less than or equal to start_time.
    """
    if end_time <= start_time:
        raise ValueError("End time must be greater than start time")

    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-i', str(input_video_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # Use stream copy for faster processing
        str(output_video_path),
        '-y'  # Overwrite output file if it exists
    ]
    
    try:
        logger.info(f"Trimming video from {start_time}s to {end_time}s")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully saved trimmed video to {output_video_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to trim video: {e.stderr}")
        raise


def trim_video_command(
    input_path: Path = typer.Argument(..., help="Input video file path", exists=True),
    output_path: Path = typer.Argument(..., help="Output video file path"),
    start_time: float = typer.Argument(..., help="Start time in seconds"),
    end_time: float = typer.Argument(..., help="End time in seconds"),
) -> None:
    """Trim a video to a specified start and end time."""
    try:
        trim_video(input_path, output_path, start_time, end_time)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(trim_video_command) 
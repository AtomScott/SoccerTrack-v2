"""Functions for extracting video metadata."""

from pathlib import Path

import cv2
from exiftool import ExifToolHelper
from loguru import logger


def get_fps(video_path: Path) -> float | None:
    """
    Get the frames per second (FPS) of a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        float | None: The video's FPS if found, None if unable to determine FPS.
    """
    with ExifToolHelper() as et:
        metadata = et.get_metadata(str(video_path))
        fps = metadata[0].get("Video", {}).get("FrameRate") or metadata[0].get("Video", {}).get("VideoFrameRate")

        if fps is None:
            logger.warning(f"FPS not found in metadata for video: {video_path}. Trying OpenCV as fallback.")
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            else:
                logger.error(f"Failed to open video file: {video_path}")
                return None

    return fps


def get_total_frames(video_path: Path) -> int:
    """
    Get the total number of frames in a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        int: Total number of frames in the video.
        
    Raises:
        cv2.error: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise cv2.error(f"Failed to open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames 
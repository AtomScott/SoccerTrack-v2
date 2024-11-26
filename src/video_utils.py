from pathlib import Path

import cv2
from exiftool import ExifToolHelper
from loguru import logger


def get_fps(video_path: Path) -> float | None:
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
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

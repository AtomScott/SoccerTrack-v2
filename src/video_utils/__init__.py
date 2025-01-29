"""Video utility functions for processing and analyzing video files."""

from .metadata import get_fps, get_total_frames
from .trim import trim_video

__all__ = [
    "get_fps",
    "get_total_frames",
    "trim_video",
] 
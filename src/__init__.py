"""Command implementations for the CLI application."""

from .help import print_help
from .example import log_string
from .visualization.plot_coordinates_on_video import plot_coordinates_on_video
from .visualization.plot_bboxes_on_video import plot_bboxes_on_video
from .detection.yolov8 import detect_objects
from .data_association.create_ground_truth import create_ground_truth_mot

__all__ = [
    "print_help",
    "log_string",
    "plot_coordinates_on_video",
    "plot_bboxes_on_video",
    "detect_objects",
    "create_ground_truth_mot",
]

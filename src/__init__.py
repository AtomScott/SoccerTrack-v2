"""Command implementations for the CLI application."""

from .help import print_help
from .example import log_string
from .visualization.plot_coordinates_on_video import plot_coordinates_on_video
from .visualization.plot_bboxes_on_video import plot_bboxes_on_video
from .detection.yolov8 import detect_objects
from .data_utils.create_yolo_dataset import create_yolo_dataset
from .video_utils.trim_video_into_halves import trim_video_into_halves
from .coordinate_conversion.convert_raw_to_pitch_plane import convert_raw_to_pitch_plane
from .coordinate_conversion.convert_pitch_plane_to_image_plane import convert_pitch_plane_to_image_plane
from .coordinate_conversion.convert_image_plane_to_bounding_box import convert_image_plane_to_bounding_box
from .calibration.generate_calibration_mappings import generate_calibration_mappings

__all__ = [
    "print_help",
    "log_string",
    "plot_coordinates_on_video",
    "plot_bboxes_on_video",
    "detect_objects",
    "create_yolo_dataset",
    "trim_video_into_halves",
    "convert_raw_to_pitch_plane",
    "convert_pitch_plane_to_image_plane",
    "convert_image_plane_to_bounding_box",
    "generate_calibration_mappings",
    "calibrate_camera",
]

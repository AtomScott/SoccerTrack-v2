"""Command implementations for the CLI application.

Public names are exposed *lazily* via PEP 562 module ``__getattr__``: the heavy
dependency chain (loguru / cv2 / ultralytics / lightning) is only imported when a
symbol is actually accessed, not on ``import src``. This keeps lightweight entry
points — notably the evaluation modules (``src.evaluation.bas_map`` /
``gs_hota`` / ``mot_hota``) — importable even when those heavy deps are missing
or broken.
"""

from __future__ import annotations

from importlib import import_module

# Public name -> "submodule:attr". Imported on first attribute access.
_LAZY_EXPORTS: dict[str, str] = {
    "print_help": "src.help:print_help",
    "log_string": "src.example:log_string",
    "plot_coordinates_on_video": "src.visualization.plot_coordinates_on_video:plot_coordinates_on_video",
    "plot_bboxes_on_video": "src.visualization.plot_bboxes_on_video:plot_bboxes_on_video",
    "detect_objects": "src.detection.yolov8:detect_objects",
    # Moved out of data_utils into a dedicated src/yolo module.
    "create_yolo_dataset": "src.yolo:create_yolo_dataset",
    "train_yolo_model": "src.yolo:train_yolo_model",
    "trim_video_into_halves": "src.video_utils.trim_video_into_halves:trim_video_into_halves",
    "convert_raw_to_pitch_plane": "src.coordinate_conversion.convert_raw_to_pitch_plane:convert_raw_to_pitch_plane",
    "convert_pitch_plane_to_image_plane": (
        "src.coordinate_conversion.convert_pitch_plane_to_image_plane:convert_pitch_plane_to_image_plane"
    ),
    "convert_image_plane_to_bounding_box": (
        "src.coordinate_conversion.convert_image_plane_to_bounding_box:convert_image_plane_to_bounding_box"
    ),
    "generate_calibration_mappings": "src.calibration.generate_calibration_mappings:generate_calibration_mappings",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Lazily resolve a public symbol (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value  # cache so subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

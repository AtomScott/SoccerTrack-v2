"""
YOLOv8 detection module for SoccerTrack-V2.
Handles video processing and object detection using Ultralytics YOLO.
"""

import tempfile
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from ..video_utils import get_total_frames


class DetectionResult(dict):
    """Type definition for detection results."""

    frame: int
    id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float
    x: float
    y: float
    z: float
    class_name: str


def detect_objects(
    match_id: str,
    video_path: Path | str,
    output_path: Path | str,
    weights_path: Path | str,
    tracker_config: dict | DictConfig,
    event_period: str | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    vid_stride: int = 1,
) -> None:
    """Run YOLOv8 inference on a single video.

    Args:
        match_id: The ID of the match
        video_path: Path to input video
        output_path: Path to save detection results
        weights_path: Path to YOLOv8 weights file
        tracker_config: Configuration for the tracker
        event_period: Event period (FIRST_HALF or SECOND_HALF)
        conf: Confidence threshold
        iou: IOU threshold
        imgsz: Input image size
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    weights_path = Path(weights_path)

    # Validate inputs
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if event_period and event_period not in ["FIRST_HALF", "SECOND_HALF"]:
        raise ValueError("event_period must be either FIRST_HALF or SECOND_HALF")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        logger.info(f"Loading YOLO model from {weights_path}")
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

    logger.info(f"Processing video: {video_path}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp_tracker:
        # Create temporary tracker config
        if isinstance(tracker_config, dict):
            yaml.dump(tracker_config, tmp_tracker, default_flow_style=False)
        elif isinstance(tracker_config, DictConfig):
            OmegaConf.save(tracker_config, tmp_tracker.name)
        else:
            logger.error(f"Invalid tracker config type: {type(tracker_config)}")
            raise ValueError("Invalid tracker config type")
        tmp_tracker.flush()

        # Run detection using config parameters
        detections: list[DetectionResult] = []
        try:
            results = model.track(
                source=str(video_path),
                tracker=tmp_tracker.name,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                stream=True,
                vid_stride=vid_stride,
            )

            total_frames = get_total_frames(video_path)
            for frame_idx, res in enumerate(
                tqdm(results, desc=f"Processing {video_path.stem}", total=total_frames // vid_stride)
            ):
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        {
                            "frame": frame_idx,
                            "id": int(box.id[0]) if box.id is not None else -1,
                            "bb_left": x1,
                            "bb_top": y1,
                            "bb_width": x2 - x1,
                            "bb_height": y2 - y1,
                            "conf": float(box.conf[0]),
                            "x": -1,
                            "y": -1,
                            "z": -1,
                            "class_name": res.names[int(box.cls[0])],
                        }
                    )

            # Create DataFrame and save results
            df = pd.DataFrame(detections)
            mot_columns = [
                "frame",
                "id",
                "bb_left",
                "bb_top",
                "bb_width",
                "bb_height",
                "conf",
                "x",
                "y",
                "z",
                "class_name",
            ]
            df = df[mot_columns]
            df[["frame", "id"]] = df[["frame", "id"]].astype(int)
            df.to_csv(output_path, index=False, header=False)

            logger.success(f"Processed {video_path}. Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {e}")
            raise

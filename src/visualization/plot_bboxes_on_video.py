"""Core functionality for plotting bounding boxes on video frames."""

import json
from pathlib import Path
from typing import Dict
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from deffcode import FFdecoder
from loguru import logger
from tqdm import tqdm
from vidgear.gears import WriteGear

from ..video_utils import get_fps


def load_detections(match_id: str, detections_path: Path) -> pd.DataFrame:
    """
    Load detection results from CSV.

    Args:
        match_id (str): The ID of the match.
        detections_path (Path): Path to the CSV file containing detections.

    Returns:
        pd.DataFrame: Detections dataframe.
    """
    logger.info(f"Loading detections from: {detections_path}")
    detections = pd.read_csv(
        detections_path,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z", "class_name"],
    )
    detections["frame"] = detections["frame"].astype(int)
    return detections


def plot_frame_detections(
    frame: np.ndarray,
    detections: pd.DataFrame,
    team_colors: Dict[str, list],
    track_colors: defaultdict,
    show_ids: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Plot detections for a single frame onto the image.

    Args:
        frame (np.ndarray): The video frame image.
        detections (pd.DataFrame): Detections to plot.
        team_colors (Dict[str, list]): Dictionary containing RGB colors for teams.
        track_colors (defaultdict): Dictionary containing RGB colors for track IDs.
        show_ids (bool): Whether to display track IDs on the bounding boxes.
        line_thickness (int): Thickness of bounding box lines.
        font_scale (float): Scale of font for labels.

    Returns:
        np.ndarray: Annotated frame with plotted detections.
    """
    frame_with_boxes = frame.copy()

    for _, det in detections.iterrows():
        x1, y1 = int(det["bb_left"]), int(det["bb_top"])
        x2 = x1 + int(det["bb_width"])
        y2 = y1 + int(det["bb_height"])

        # Choose color based on whether we're showing IDs
        if show_ids:
            # Use team colors for bounding boxes when showing IDs
            box_color = tuple(team_colors.get(det["class_name"], [128, 128, 128]))  # Gray as fallback
            label_color = tuple(track_colors[str(det["id"])])  # Unique color for each ID
        else:
            # Use unique colors for each player when not showing IDs
            box_color = tuple(track_colors[str(det["id"])])  # Unique color for each player

        # Draw bounding box
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, line_thickness)

        # Add label with ID if enabled
        if show_ids:
            label = f"{det['id']}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)[0]
            cv2.rectangle(
                frame_with_boxes,
                (x1, y1 - text_size[1] - 4),
                (x1 + text_size[0], y1),
                label_color,
                -1,
            )
            cv2.putText(
                frame_with_boxes,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
                cv2.LINE_AA,
            )

    return frame_with_boxes


def plot_bboxes_on_video(
    match_id: str,
    video_path: Path | str,
    detections_path: Path | str,
    output_path: Path | str | None = None,
    first_frame_only: bool = False,
    team_colors: Dict[str, list] | None = None,
    track_colors: Dict[str, list] | None = None,
    show_ids: bool = True,
    default_output_folder: Path | str = Path("output"),
    config_path: Path | str | None = None,
    classes: list[str] | None = None,
) -> None:
    """
    Process video frames and plot bounding boxes.

    Args:
        match_id (str): The ID of the match.
        video_path (Path | str): Path to the input video file.
        detections_path (Path | str): Path to the detections CSV file.
        output_path (Path | str | None): Path to save the output video/image.
        first_frame_only (bool): Only process the first frame and save as image.
        team_colors (Dict[str, list] | None): Dictionary containing RGB colors for teams.
        track_colors (Dict[str, list] | None): Dictionary containing RGB colors for track IDs.
        show_ids (bool): Whether to display track IDs on the bounding boxes.
        default_output_folder (Path | str): Default folder for output if output_path is not specified.
        config_path (Path | str | None): Path to config file.
        classes (list[str] | None): List of classes to visualize. If None, show all classes.
    """
    # Convert paths to Path objects
    video_path = Path(video_path)
    detections_path = Path(detections_path)
    default_output_folder = Path(default_output_folder)

    # Set up base colors for teams
    base_team_colors = {
        "ball": [255, 0, 0],  # Red for ball
        "9701": [0, 0, 255],  # Blue for team 1
        "9834": [0, 255, 0],  # Green for team 2
    }
    team_colors = team_colors or base_team_colors

    # Set up colors for track IDs with defaultdict
    # Use HSV color space for better color distribution when not showing IDs
    def generate_unique_color():
        if not show_ids:
            # Generate colors in HSV space for better distribution
            hue = np.random.randint(0, 180)  # Hue range in OpenCV is [0, 180]
            saturation = np.random.randint(150, 256)  # High saturation for vivid colors
            value = np.random.randint(150, 256)  # High value for visibility
            # Convert to BGR
            bgr_color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
            return bgr_color.tolist()
        else:
            # Original random RGB colors for ID labels
            return [np.random.randint(0, 255) for _ in range(3)]

    track_colors = defaultdict(generate_unique_color, track_colors or {})

    # Setup output path
    if not output_path:
        output_folder = default_output_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / (
            f"{match_id}_bbox_annotated.jpg" if first_frame_only else f"{match_id}_bbox_annotated.mp4"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")

    # Load detections and filter by class if specified
    detections_df = load_detections(match_id, detections_path)
    if classes is not None:
        logger.info(f"Filtering detections for classes: {classes}")
        detections_df = detections_df[detections_df["class_name"].isin(classes)]
        if detections_df.empty:
            logger.warning("No detections found for specified classes!")

    # Initialize video decoder
    logger.info(f"Decoding video from {video_path}")
    assert video_path.exists(), f"Video file does not exist: {video_path}"
    decoder = FFdecoder(str(video_path), frame_format="bgr24").formulate()

    if first_frame_only:
        try:
            frame = next(decoder.generateFrame())
            if frame is not None:
                frame_num = detections_df["frame"].min()
                frame_dets = detections_df[detections_df["frame"] == frame_num]
                annotated_frame = plot_frame_detections(frame, frame_dets, team_colors, track_colors, show_ids)
                cv2.imwrite(str(output_path), annotated_frame)
                logger.info(f"Saved annotated frame to {output_path}")
        except StopIteration:
            logger.error("No frames found in the video.")
        finally:
            decoder.terminate()
    else:
        input_framerate = get_fps(video_path)

        output_params = {
            "-input_framerate": input_framerate,
            "-vcodec": "libx264",
        }
        writer = WriteGear(output=f"{output_path}", logging=True, **output_params)

        try:
            for frame_num, frame in enumerate(tqdm(decoder.generateFrame(), desc="Processing frames")):
                if frame is None:
                    logger.warning(f"Frame {frame_num} is None. Skipping.")
                    continue
                frame_dets = detections_df[detections_df["frame"] == frame_num]
                if not frame_dets.empty:
                    annotated_frame = plot_frame_detections(frame, frame_dets, team_colors, track_colors, show_ids)
                    writer.write(annotated_frame)
                else:
                    writer.write(frame)
        except Exception as e:
            logger.error(f"An error occurred while processing frames: {e}")
            raise e
        finally:
            writer.close()
            decoder.terminate()
            logger.info(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot bounding boxes on video frames")
    parser.add_argument("--match_id", type=str, required=True, help="Match ID")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--detections_path", type=str, required=True, help="Path to detections CSV")
    parser.add_argument("--output_path", type=str, help="Path to save output video/image")
    parser.add_argument("--first_frame_only", action="store_true", help="Only process first frame")
    parser.add_argument("--show_ids", action="store_true", default=True, help="Show track IDs on bounding boxes")
    parser.add_argument("--classes", nargs="+", help="List of classes to visualize")

    args = parser.parse_args()

    plot_bboxes_on_video(
        match_id=args.match_id,
        video_path=args.video_path,
        detections_path=args.detections_path,
        output_path=args.output_path,
        first_frame_only=args.first_frame_only,
        show_ids=args.show_ids,
        classes=args.classes,
    )

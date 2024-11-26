"""Core functionality for plotting coordinates on video frames."""

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from deffcode import FFdecoder
from loguru import logger
from tqdm import tqdm
from vidgear.gears import WriteGear


def load_coordinates(match_id: str, coordinates_path: Path, event_period: str | None = None) -> pd.DataFrame:
    """
    Load pitch plane coordinates from CSV.

    Args:
        match_id (str): The ID of the match.
        coordinates_path (Path): Path to the CSV file containing coordinates.
        event_period (str | None): Event period to filter coordinates (e.g., 'FIRST_HALF').

    Returns:
        pd.DataFrame: Filtered coordinates dataframe.
    """
    logger.info(f"Loading coordinates from: {coordinates_path}")
    coordinates = pd.read_csv(coordinates_path)
    coordinates["frame"] = coordinates["frame"].astype(int) - coordinates["frame"].min()

    if event_period:
        logger.info(f"Filtering coordinates for event period: {event_period}")
        coordinates = coordinates[coordinates["event_period"] == event_period]

    return coordinates


def plot_frame_coordinates(
    frame: np.ndarray,
    coordinates: pd.DataFrame,
    H: np.ndarray,
    colors: Dict[str, list],
    point_sizes: Dict[str, int],
    pitch_length: float,
    pitch_width: float,
) -> np.ndarray:
    """
    Plot coordinates for a single frame onto the image.

    Args:
        frame (np.ndarray): The video frame image.
        coordinates (pd.DataFrame): Coordinates to plot.
        H (np.ndarray): Homography matrix.
        colors (Dict[str, list]): Dictionary containing RGB colors for each team and ball.
        point_sizes (Dict[str, int]): Dictionary containing point sizes for players and ball.
        pitch_length (float): Length of the pitch in meters.
        pitch_width (float): Width of the pitch in meters.

    Returns:
        np.ndarray: Annotated frame with plotted coordinates.
    """
    frame_with_points = frame.copy()

    for _, row in coordinates.iterrows():
        point = np.array([[row["x"] * pitch_length, row["y"] * pitch_width]], dtype=np.float32)
        point_reshaped = point.reshape(-1, 1, 2)

        projected_point = cv2.perspectiveTransform(point_reshaped, H)
        x, y = projected_point[0][0]

        if row["id"] == "ball":
            color = tuple(colors["ball"])
            size = point_sizes["ball"]
        else:
            team_id = int(row["teamId"])
            color = tuple(colors.get(str(team_id), [255, 255, 255]))  # Ensure team_id is string key
            size = point_sizes["player"]

        cv2.circle(frame_with_points, (int(x), int(y)), size, color, -1)

    return frame_with_points


def plot_coordinates_on_video(
    match_id: str,
    video_path: Path | str,
    homography_path: Path | str,
    coordinates_path: Path | str,
    output_path: Path | str | None = None,
    first_frame_only: bool = False,
    event_period: str | None = None,
    colors: Dict[str, list] | None = None,
    point_sizes: Dict[str, int] | None = None,
    default_output_folder: Path | str = Path("output"),
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """
    Process video frames and plot coordinates.

    Args:
        match_id (str): The ID of the match.
        video_path (Path | str): Path to the input video file.
        homography_path (Path | str): Path to the homography matrix file.
        coordinates_path (Path | str): Path to the coordinates CSV file.
        output_path (Path | str | None): Path to save the output video/image.
        first_frame_only (bool): Only process the first frame and save as image.
        event_period (str | None): Event period to filter coordinates.
        colors (Dict[str, list] | None): Dictionary containing RGB colors for each team and ball.
        point_sizes (Dict[str, int] | None): Dictionary containing point sizes for players and ball.
        default_output_folder (Path | str): Default folder for output if output_path is not specified.
        pitch_length (float): Length of the pitch in meters.
        pitch_width (float): Width of the pitch in meters.
    """
    # Convert paths to Path objects
    video_path = Path(video_path)
    homography_path = Path(homography_path)
    coordinates_path = Path(coordinates_path)
    default_output_folder = Path(default_output_folder)

    # Set default values for colors and point sizes if not provided
    if colors is None:
        colors = {
            "ball": [255, 0, 0],  # Red
            "1": [0, 0, 255],  # Blue
            "2": [0, 255, 0],  # Green
        }
    if point_sizes is None:
        point_sizes = {"ball": 3, "player": 5}

    # Setup output path
    if not output_path:
        output_folder = default_output_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / (f"{match_id}_annotated.jpg" if first_frame_only else f"{match_id}_annotated.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")

    # Load coordinates with optional event_period filtering
    coordinates_df = load_coordinates(match_id, coordinates_path, event_period=event_period)

    # Load homography matrix
    logger.info(f"Loading homography matrix from: {homography_path}")
    H = np.load(homography_path)

    # Initialize video decoder
    decoder = FFdecoder(str(video_path), frame_format="bgr24").formulate()

    if first_frame_only:
        try:
            frame = next(decoder.generateFrame())
            if frame is not None:
                frame_num = coordinates_df["frame"].min()
                frame_coords = coordinates_df[coordinates_df["frame"] == frame_num]
                annotated_frame = plot_frame_coordinates(
                    frame, frame_coords, H, colors, point_sizes, pitch_length, pitch_width
                )
                cv2.imwrite(str(output_path), annotated_frame)
                logger.info(f"Saved annotated frame to {output_path}")
        except StopIteration:
            logger.error("No frames found in the video.")
        finally:
            decoder.terminate()
    else:
        try:
            with open(decoder.metadata, "r") as meta_file:
                metadata = json.load(meta_file)
            input_framerate = metadata.get("output_framerate", 30)  # Default to 30 if not found
        except Exception as e:
            logger.error(f"Failed to load decoder metadata: {e}")
            input_framerate = 30  # Default fallback

        output_params = {
            "-input_framerate": input_framerate,
            "-vcodec": "libx264",
        }
        writer = WriteGear(output=f"file://{output_path}", logging=True, **output_params)

        try:
            for frame_num, frame in enumerate(tqdm(decoder.generateFrame(), desc="Processing frames")):
                if frame is None:
                    logger.warning(f"Frame {frame_num} is None. Skipping.")
                    continue
                frame_coords = coordinates_df[coordinates_df["frame"] == frame_num]
                if not frame_coords.empty:
                    annotated_frame = plot_frame_coordinates(
                        frame, frame_coords, H, colors, point_sizes, pitch_length, pitch_width
                    )
                    writer.write(annotated_frame)
                else:
                    writer.write(frame)
        except Exception as e:
            logger.error(f"An error occurred while processing frames: {e}")
        finally:
            writer.close()
            decoder.terminate()
            logger.info(f"Saved annotated video to {output_path}")

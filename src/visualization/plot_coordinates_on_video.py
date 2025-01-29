"""Plot coordinates on video frames using pre-computed image plane coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deffcode import FFdecoder
from loguru import logger
from tqdm import tqdm
from vidgear.gears import WriteGear


def plot_frame_coordinates(
    frame: np.ndarray,
    coordinates: pd.DataFrame,
    colors: dict[str, list],
    point_sizes: dict[str, int],
) -> np.ndarray:
    """
    Plot coordinates for a single frame onto the image.

    Args:
        frame (np.ndarray): The video frame image.
        coordinates (pd.DataFrame): Image plane coordinates to plot.
        colors (Dict[str, list]): Dictionary containing RGB colors for each team and ball.
        point_sizes (Dict[str, int]): Dictionary containing point sizes for players and ball.

    Returns:
        np.ndarray: Annotated frame with plotted coordinates.
    """
    frame_with_points = frame.copy()

    for _, row in coordinates.iterrows():
        x, y = int(row["x"]), int(row["y"])

        if row["id"] == "ball":
            color = tuple(colors.get("ball", [255, 0, 0]))  # Red as fallback
            size = point_sizes.get("ball", 3)
        else:
            team_id = str(int(row["teamId"]))
            color = tuple(colors.get(team_id, [255, 255, 255]))  # White as fallback
            size = point_sizes.get("player", 5)

        cv2.circle(frame_with_points, (x, y), size, color, -1)

    return frame_with_points


def plot_coordinates_on_video(
    match_id: str,
    video_path: Path | str,
    coordinates_path: Path | str,
    output_path: Path | str | None = None,
    first_frame_only: bool = False,
    colors: dict[str, list] | None = None,
    point_sizes: dict[str, int] | None = None,
    default_output_folder: Path | str = Path("output"),
) -> None:
    """
    Process video frames and plot coordinates.

    Args:
        match_id (str): The ID of the match.
        video_path (Path | str): Path to the input video file.
        coordinates_path (Path | str): Path to the image plane coordinates CSV file.
        output_path (Path | str | None): Path to save the output video/image.
        first_frame_only (bool): Only process the first frame and save as image.
        colors (Dict[str, list] | None): Dictionary containing RGB colors for each team and ball.
        point_sizes (Dict[str, int] | None): Dictionary containing point sizes for players and ball.
        default_output_folder (Path | str): Default folder for output if output_path is not specified.
    """
    # Convert paths to Path objects
    video_path = Path(video_path)
    coordinates_path = Path(coordinates_path)
    default_output_folder = Path(default_output_folder)

    # Set default values for colors and point sizes if not provided
    if colors is None:
        colors = {
            "ball": [255, 0, 0],  # Red
            "9701": [0, 0, 255],  # Blue
            "9834": [0, 255, 0],  # Green
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

    # Load coordinates
    logger.info(f"Loading coordinates from: {coordinates_path}")
    coordinates_df = pd.read_csv(coordinates_path)

    # Initialize video decoder
    decoder = FFdecoder(str(video_path.resolve()), frame_format="bgr24").formulate()

    if first_frame_only:
        try:
            frame = next(decoder.generateFrame())
            if frame is not None:
                frame_num = coordinates_df["frame"].min()
                frame_coords = coordinates_df[coordinates_df["frame"] == frame_num]
                annotated_frame = plot_frame_coordinates(frame, frame_coords, colors, point_sizes)
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
        writer = WriteGear(output=str(output_path.resolve()), logging=True, **output_params)

        try:
            for frame_num, frame in enumerate(tqdm(decoder.generateFrame(), desc="Processing frames")):
                if frame is None:
                    logger.warning(f"Frame {frame_num} is None. Skipping.")
                    continue
                frame_coords = coordinates_df[coordinates_df["frame"] == frame_num]
                if not frame_coords.empty:
                    annotated_frame = plot_frame_coordinates(frame, frame_coords, colors, point_sizes)
                    writer.write(annotated_frame)
                else:
                    writer.write(frame)
        except Exception as e:
            logger.error(f"An error occurred while processing frames: {e}")
        finally:
            writer.close()
            decoder.terminate()
            logger.info(f"Saved annotated video to {output_path}")

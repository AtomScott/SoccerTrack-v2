"""Command to create ground truth MOT file with fixed-size bounding boxes."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from src.csv_utils import load_coordinates


def create_ground_truth_mot_from_coordinates(
    match_id: str,
    coordinates_path: Path | str,
    homography_path: Path | str,
    output_path: Path | str,
    event_period: str | None = None,
    bb_width: float = 100.0,  # Changed to match our previous fixed size
    bb_height: float = 20.0,  # Changed to match our previous fixed size
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """
    Create a ground truth MOT-like CSV purely from the given coordinates.
    Each (x, y) point is converted to a fixed-size bounding box,
    with (x, y) as the bottom center of the box.

    Args:
        match_id (str): The ID of the match to filter on.
        coordinates_path (Path | str): Path to the CSV file containing coordinates.
        homography_path (Path | str): Path to the homography matrix file.
        output_path (Path | str): Where to save the MOT-like output CSV file.
        event_period (str, optional): If provided, filter coordinates by event period. Defaults to None.
        bb_width (float, optional): Desired bounding box width. Defaults to 100.0.
        bb_height (float, optional): Desired bounding box height. Defaults to 20.0.
        pitch_length (float, optional): Length of the pitch in meters. Defaults to 105.0.
        pitch_width (float, optional): Width of the pitch in meters. Defaults to 68.0.
    """
    logger.info("Loading coordinates")
    coordinates_df = load_coordinates(Path(coordinates_path), match_id, event_period)

    # Load homography matrix
    logger.info(f"Loading homography matrix from {homography_path}")
    H = np.load(homography_path)

    # Exclude ball if present
    coordinates_df = coordinates_df[coordinates_df["id"] != "ball"]
    logger.info(f"Unique coordinate IDs: {coordinates_df['teamId'].unique()}")

    logger.info("Transforming coordinates into bounding boxes")

    # Copy coordinates to a new DataFrame
    mot_df = coordinates_df.copy()

    # Convert pitch coordinates to image coordinates
    # First scale the coordinates to meters
    pitch_coords = mot_df[["x", "y"]].values
    pitch_coords[:, 0] *= pitch_length
    pitch_coords[:, 1] *= pitch_width

    # Reshape for cv2.perspectiveTransform
    pitch_coords = pitch_coords.reshape(-1, 1, 2).astype(np.float32)

    # Transform to image coordinates
    image_coords = cv2.perspectiveTransform(pitch_coords, H)
    image_coords = image_coords.reshape(-1, 2)

    # Update coordinates in DataFrame
    mot_df["x"] = image_coords[:, 0]
    mot_df["y"] = image_coords[:, 1]

    # Define new columns for bounding box
    mot_df["bb_width"] = bb_width
    mot_df["bb_height"] = bb_height

    # Bottom center at (x, y):
    # bb_left = x - bb_width/2
    # bb_top = y - bb_height
    mot_df["bb_left"] = mot_df["x"] - (bb_width / 2.0)
    mot_df["bb_top"] = mot_df["y"] - bb_height

    # Add confidence column (required for MOT format)
    mot_df["conf"] = 1.0
    mot_df["z"] = 0.0  # Add Z coordinate as 0
    mot_df["class_name"] = "person"  # Add class name

    # Order columns to match MOT format
    desired_cols = [
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
    # Only keep columns that exist in mot_df
    final_cols = [c for c in desired_cols if c in mot_df.columns]
    mot_df = mot_df[final_cols]

    # Sort by frame then id for a clean format
    mot_df = mot_df.sort_values(["frame", "id"])

    logger.info(f"Saving MOT DataFrame to {output_path}")
    # Save without headers for MOT format
    mot_df.to_csv(output_path, index=False, header=False)
    logger.info("Finished creating ground truth MOT from coordinates.")


if __name__ == "__main__":
    # This allows the command to be run directly for testing
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Create ground truth MOT file from coordinates only.")
    parser.add_argument(
        "--config_path", type=str, default="configs/default_config.yaml", help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    create_ground_truth_mot_from_coordinates(**config.create_ground_truth_mot_from_coordinates)

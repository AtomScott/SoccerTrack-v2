"""Command to create ground truth MOT file with fixed-size bounding boxes."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from src.csv_utils import load_coordinates
from src.data_association.analyze_bbox_dimensions import load_bbox_models, estimate_bbox_dimensions


def remove_duplicates_and_linear_interpolate(df, group_col="id", frame_col="frame", interp_cols=("x", "y"), period=30):
    """
    For each group (by `group_col`), any row whose `frame` is NOT a multiple
    of `period` has its `interp_cols` set to NaN. Then those NaNs are filled
    by linear interpolation. The rest of the columns are left as-is.
    """
    # Make sure the DataFrame is sorted by group_col, then by frame_col
    df = df.sort_values(by=[group_col, frame_col]).copy()

    # Group by the chosen column (e.g., player or ball ID)
    grouped = df.groupby(group_col, as_index=False)

    def process_group(group):
        # Identify rows whose frame is a multiple of period
        mask = group[frame_col] % period == 0

        # Set values in x/y columns to NaN if not multiple of period
        for col in interp_cols:
            # Use .loc to avoid chained assignments
            group.loc[~mask, col] = np.nan

        # Now interpolate the NaNs in those columns
        group[list(interp_cols)] = group[list(interp_cols)].interpolate(method="linear")

        # Return the group as-is (no extra index renaming)
        return group

    # Apply the process to each group
    # `include_groups=False` avoids the deprecation warning in pandas >= 2.0
    result = grouped.apply(process_group, include_groups=True)

    # Reset the index from groupby/apply
    result.reset_index(drop=True, inplace=True)

    # Finally, sort by frame if desired
    result.sort_values(by=[frame_col], inplace=True)

    return result


def create_ground_truth_mot_from_coordinates(
    match_id: str,
    coordinates_path: Path | str,
    homography_path: Path | str,
    output_path: Path | str,
    bbox_models_path: Path | str | None = None,
    event_period: str | None = None,
    bb_width: float = 5.0,
    bb_height: float = 20.0,
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
        bbox_models_path (Path | str | None): Path to saved bbox regression models. If provided, uses dynamic bbox sizes.
        event_period (str, optional): If provided, filter coordinates by event period. Defaults to None.
        bb_width (float, optional): Default bounding box width if no models provided. Defaults to 5.0.
        bb_height (float, optional): Default bounding box height if no models provided. Defaults to 20.0.
        pitch_length (float, optional): Length of the pitch in meters. Defaults to 105.0.
        pitch_width (float, optional): Width of the pitch in meters. Defaults to 68.0.
    """
    logger.info("Loading coordinates")
    coordinates_df = load_coordinates(Path(coordinates_path), match_id, event_period)

    # Load homography matrix
    logger.info(f"Loading homography matrix from {homography_path}")
    H = np.load(homography_path)

    # Load bbox models if provided
    bbox_models = None
    if bbox_models_path:
        logger.info(f"Loading bbox regression models from {bbox_models_path}")
        width_model, height_model, ranges = load_bbox_models(bbox_models_path)
        bbox_models = (width_model, height_model, ranges)

    # Exclude ball if present
    coordinates_df = coordinates_df[coordinates_df["id"] != "ball"]
    coordinates_df = remove_duplicates_and_linear_interpolate(coordinates_df, "id", "frame", ("x", "y"), 30)
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

    # Define bounding box dimensions
    if bbox_models:
        logger.info("Using regression models for dynamic bbox dimensions")
        width_model, height_model, ranges = bbox_models
        dimensions = []
        total_frames = len(mot_df)
        for i, (x, y) in tqdm(enumerate(zip(mot_df["x"], mot_df["y"])), total=total_frames):
            dimensions.append(estimate_bbox_dimensions(x, y, width_model, height_model, ranges))
        mot_df["bb_width"], mot_df["bb_height"] = zip(*dimensions)
    else:
        logger.info(f"Using fixed bbox dimensions: width={bb_width}, height={bb_height}")
        mot_df["bb_width"] = bb_width
        mot_df["bb_height"] = bb_height

    # Bottom center at (x, y):
    # bb_left = x - bb_width/2
    # bb_top = y - bb_height
    mot_df["bb_left"] = mot_df["x"] - (mot_df["bb_width"] / 2.0)
    mot_df["bb_top"] = mot_df["y"] - mot_df["bb_height"]

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

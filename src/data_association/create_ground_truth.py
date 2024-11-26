"""Command to create ground truth MOT file by associating bounding boxes with coordinates."""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from time import sleep
from src.csv_utils import load_coordinates, load_detections


def interpolate_tracks(df: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """
    Interpolate missing detections in tracks.

    Args:
        df (pd.DataFrame): DataFrame with tracking data.
        max_gap (int, optional): Maximum number of frames to interpolate between existing detections. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with interpolated tracks.
    """
    logger.info("Interpolating tracks")
    interpolated_df = []

    for track_id in df["id"].unique():
        track = df[df["id"] == track_id].sort_values("frame")

        # Find frame gaps
        frame_diff = track["frame"].diff()
        has_gaps = (frame_diff > 1).any()

        if has_gaps:
            # Create continuous frame sequence
            full_frames = pd.DataFrame({"frame": range(track["frame"].min(), track["frame"].max() + 1)})
            track = pd.merge(full_frames, track, on="frame", how="left")

            # Interpolate all numeric columns
            numeric_cols = ["bb_left", "bb_top", "bb_width", "bb_height", "x", "y", "conf"]
            for col in numeric_cols:
                track[col] = track[col].interpolate(method="linear", limit=max_gap)

            # Fill non-numeric columns
            track["id"] = track_id

            # Remove rows that are still NaN after interpolation
            track = track.dropna(subset=numeric_cols)

        interpolated_df.append(track)

    result = pd.concat(interpolated_df, ignore_index=True)
    result = result.sort_values(["frame", "id"])
    logger.info(f"Interpolation complete. Tracks increased from {len(df)} to {len(result)} entries")
    return result


def preprocess_detections(df: pd.DataFrame, conf_threshold: float = 0.3) -> pd.DataFrame:
    """
    Preprocess detection data by filtering low confidence detections and non-person classes.

    Args:
        df (pd.DataFrame): DataFrame with detection data.
        conf_threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.3.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    logger.info(f"Preprocessing detections with confidence threshold {conf_threshold}")

    # TODO: Add team classification when available
    # Filter by confidence and class
    filtered_df = df[(df["conf"] > conf_threshold) & (df["class_name"] == "person")].copy()

    logger.info(f"Filtered {len(df) - len(filtered_df)} low confidence or non-person detections")
    logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
    return filtered_df


def associate_tracklets_to_coordinates(
    tracklets_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    projected_coords: np.ndarray,
    max_distance: float = 50.0,
) -> pd.DataFrame:
    """
    Associate tracklet IDs with coordinate IDs using frame-by-frame linear assignment.

    Args:
        tracklets_df: DataFrame containing detection tracklets
        coordinates_df: DataFrame containing ground truth coordinates
        projected_coords: Ground truth coordinates projected to image space
        max_distance: Maximum distance threshold for valid associations

    Returns:
        DataFrame with tracklets matched to coordinate IDs
    """
    logger.info("Starting frame-by-frame tracklet-to-coordinate association using linear assignment")

    # Track assignment counts
    # Structure: tracklet_id -> {coord_id -> count}
    assignment_counts: dict[int, dict[int, int]] = {}

    # Process each frame
    for frame in tqdm(sorted(tracklets_df["frame"].unique()), desc="Processing frames"):
        frame_dets = tracklets_df[tracklets_df["frame"] == frame]
        frame_coords = coordinates_df[coordinates_df["frame"] == frame]

        if frame_dets.empty or frame_coords.empty:
            continue

        # Calculate detection centers (bottom middle point)
        det_centers = frame_dets[["bb_left", "bb_top", "bb_width", "bb_height"]].values
        det_centers = det_centers[:, :2] + np.column_stack([det_centers[:, 2] / 2, det_centers[:, 3]])

        # Get projected coordinates for this frame
        frame_proj_coords = projected_coords[frame_coords.index]

        # Create cost matrix
        cost_matrix = np.zeros((len(frame_dets), len(frame_coords)))
        for i, det_center in enumerate(det_centers):
            for j, coord_pos in enumerate(frame_proj_coords):
                distance = np.linalg.norm(det_center - coord_pos)
                cost_matrix[i, j] = distance if distance < max_distance else 1e6

        # Perform linear assignment
        det_indices, coord_indices = linear_sum_assignment(cost_matrix)

        # Record valid assignments
        for det_idx, coord_idx in zip(det_indices, coord_indices):
            if cost_matrix[det_idx, coord_idx] < max_distance:
                tracklet_id = frame_dets.iloc[det_idx]["id"]
                coord_id = frame_coords.iloc[coord_idx]["id"]

                if tracklet_id not in assignment_counts:
                    assignment_counts[tracklet_id] = {}
                assignment_counts[tracklet_id][coord_id] = assignment_counts[tracklet_id].get(coord_id, 0) + 1

    # Create final mapping based on majority voting
    tracklet_to_coord: dict[int, int] = {}
    for tracklet_id, coord_counts in assignment_counts.items():
        if coord_counts:  # Only process if there are any assignments
            # Get coordinate ID with maximum count
            coord_id = max(coord_counts.items(), key=lambda x: x[1])[0]
            tracklet_to_coord[tracklet_id] = coord_id
            logger.info(f"Tracklet {tracklet_id} assigned to coordinate {coord_id}")

    # Update tracklet IDs
    updated_tracklets = tracklets_df.copy()
    updated_tracklets = updated_tracklets[updated_tracklets["id"].isin(tracklet_to_coord.keys())].copy()
    updated_tracklets["id"] = updated_tracklets["id"].map(tracklet_to_coord).astype(int)

    # Log statistics
    logger.info(f"Total unique tracklets before merging: {len(tracklets_df['id'].unique())}")
    logger.info(f"Total unique coordinates: {len(coordinates_df['id'].unique())}")
    logger.info(f"Total unique tracklets after merging: {len(updated_tracklets['id'].unique())}")
    logger.info(f"Total associations made: {len(tracklet_to_coord)}")
    # logger.info(f"Unique tracklet IDs: {updated_tracklets['id'].unique()}")

    return updated_tracklets


def calculate_reprojection_error(
    detections_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    projected_coords: np.ndarray,
) -> tuple[float, Dict[int, float]]:
    """
    Calculate reprojection error between bounding box centers and projected coordinates.

    Args:
        detections_df (pd.DataFrame): DataFrame with detection data.
        coordinates_df (pd.DataFrame): DataFrame with coordinate data.
        projected_coords (np.ndarray): Coordinates projected to image space.

    Returns:
        tuple[float, Dict[int, float]]: Average error and per-track errors.
    """
    errors: Dict[int, float] = {}
    all_errors = []

    # Reverse the associations from detections to coordinates
    tracklet_to_coord = {tid: cid for cid, tid in zip(detections_df["id"], detections_df["id"])}

    for tracklet_id in detections_df["id"].unique():
        coord_id = tracklet_id
        track = detections_df[detections_df["id"] == tracklet_id]

        track_errors = []

        for _, det in track.iterrows():
            # Get detection center (bottom middle point)
            det_center = np.array([det["bb_left"] + det["bb_width"] / 2, det["bb_top"] + det["bb_height"]])

            # Get corresponding coordinate
            coord = coordinates_df[coordinates_df["id"] == coord_id]
            if coord.empty:
                continue
            coord = coord.iloc[0]

            # Get projected coordinate
            coord_idx = coord.name
            proj_coord = projected_coords[coord_idx]

            # Calculate Euclidean distance
            error = np.linalg.norm(det_center - proj_coord)
            track_errors.append(error)
            all_errors.append(error)

        if track_errors:
            errors[tracklet_id] = np.mean(track_errors)

    avg_error = np.mean(all_errors) if all_errors else float("inf")
    logger.info(f"Average reprojection error: {avg_error:.2f} pixels")

    return avg_error, errors


def iterative_association_refinement(
    detections_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    projected_coords: np.ndarray,
    max_iterations: int = 5,
    error_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Iteratively refine associations based on reprojection error.

    Args:
        detections_df (pd.DataFrame): DataFrame with detection data.
        coordinates_df (pd.DataFrame): DataFrame with coordinate data.
        projected_coords (np.ndarray): Coordinates projected to image space.
        max_iterations (int, optional): Maximum number of refinement iterations. Defaults to 5.
        error_threshold (float, optional): Maximum acceptable reprojection error in pixels. Defaults to 50.0.

    Returns:
        pd.DataFrame: Refined detections DataFrame with updated IDs.
    """
    current_detections = detections_df.copy()

    for iteration in range(max_iterations):
        logger.info(f"Starting refinement iteration {iteration + 1}")

        # Calculate current errors
        avg_error, track_errors = calculate_reprojection_error(current_detections, coordinates_df, projected_coords)

        # Find problematic associations
        bad_tracklets = {tid for tid, error in track_errors.items() if error > error_threshold}

        if not bad_tracklets:
            logger.info("No problematic associations found. Stopping refinement.")
            break

        # Remove problematic tracklet associations by setting their IDs to a new unique ID
        current_detections.loc[current_detections["id"].isin(bad_tracklets), "id"] = -1  # -1 denotes unassigned

        logger.info(f"Removed {len(bad_tracklets)} problematic associations")

        # Calculate new average error
        new_avg_error, _ = calculate_reprojection_error(current_detections, coordinates_df, projected_coords)

        if new_avg_error >= avg_error:
            logger.info("No improvement in average error. Stopping refinement.")
            break

        logger.info(f"Iteration {iteration + 1}: Average error improved from {avg_error:.2f} to {new_avg_error:.2f}")

    return current_detections


def interpolate_tracks_with_coordinates(
    detections_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    projected_coords: np.ndarray,
    max_gap: int = 10,
) -> pd.DataFrame:
    """
    Fill missing detections using coordinate information.

    Args:
        detections_df (pd.DataFrame): DataFrame with detection data.
        coordinates_df (pd.DataFrame): DataFrame with coordinate data.
        projected_coords (np.ndarray): Ground truth coordinates projected to image space.
        max_gap (int, optional): Maximum number of frames to interpolate. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with filled tracks.
    """
    logger.info("Filling tracks using coordinate information")
    filled_df = []

    # Iterate through coordinate IDs (should be 22 players)
    for coord_id in tqdm(coordinates_df["id"].unique(), desc="Processing tracks"):
        # Get coordinate and detection data for this pair
        coord_track = coordinates_df[coordinates_df["id"] == coord_id]
        track = detections_df[detections_df["id"] == coord_id].copy()

        if track.empty:
            logger.warning(f"No detections found for coord_id {coord_id}")
            continue

        # Get all frames from coordinate track
        all_frames = sorted(coord_track["frame"].unique())

        # Calculate average box size for this track
        avg_width = track["bb_width"].mean()
        avg_height = track["bb_height"].mean()

        # Create new rows for missing frames
        new_rows = []
        for frame in all_frames:
            if frame not in track["frame"].values:
                coord_row = coord_track[coord_track["frame"] == frame].iloc[0]
                coord_idx = coord_row.name
                proj_pos = projected_coords[coord_idx]

                new_rows.append(
                    {
                        "frame": frame,
                        "id": coord_id,
                        "bb_left": proj_pos[0] - avg_width / 2,
                        "bb_top": proj_pos[1] - avg_height,
                        "bb_width": avg_width,
                        "bb_height": avg_height,
                        "conf": 0.5,  # Lower confidence for filled boxes
                    }
                )

        if new_rows:
            track = pd.concat([track, pd.DataFrame(new_rows)], ignore_index=True)
            track = track.sort_values("frame")

        filled_df.append(track)

    if filled_df:
        result = pd.concat(filled_df, ignore_index=True)
        logger.info(f"Track filling complete. Tracks increased from {len(detections_df)} to {len(result)} entries")
        return result
    else:
        logger.warning("No tracks were filled. Returning original detections.")
        return detections_df


def create_ground_truth_mot(
    match_id: str,
    detections_path: Path | str,
    coordinates_path: Path | str,
    homography_path: Path | str,
    output_path: Path | str,
    event_period: Optional[str] = None,
    max_distance: float = 50.0,
    conf_threshold: float = 0.3,
    max_interpolation_gap: int = 10,
    association_window_size: int = 30,
    max_refinement_iterations: int = 5,
    error_threshold: float = 50.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """
    Create ground truth MOT file by associating bounding boxes with coordinate data.

    Args:
        match_id (str): The ID of the match.
        detections_path (Path | str): Path to the detections CSV file.
        coordinates_path (Path | str): Path to the coordinates CSV file.
        homography_path (Path | str): Path to the homography matrix file.
        output_path (Path | str): Path to save the output MOT file.
        event_period (Optional[str], optional): Event period to filter coordinates. Defaults to None.
        max_distance (float, optional): Maximum distance for matching detections with coordinates. Defaults to 50.0.
        conf_threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.3.
        max_interpolation_gap (int, optional): Maximum number of frames to interpolate. Defaults to 10.
        association_window_size (int, optional): Number of frames to consider for tracklet association. Defaults to 30.
        max_refinement_iterations (int, optional): Maximum number of refinement iterations. Defaults to 5.
        error_threshold (float, optional): Maximum acceptable reprojection error in pixels. Defaults to 50.0.
        pitch_length (float, optional): Length of the pitch in meters. Defaults to 105.0.
        pitch_width (float, optional): Width of the pitch in meters. Defaults to 68.0.
    """
    logger.info("Starting ground truth MOT creation")

    # 1. Load data
    detections_df = load_detections(Path(detections_path))
    coordinates_df = load_coordinates(Path(coordinates_path), match_id, event_period)
    H = np.load(homography_path)

    # 2. Preprocess detections
    detections_df = preprocess_detections(detections_df, conf_threshold)
    coordinates_df = coordinates_df[coordinates_df["id"] != "ball"]  # Exclude ball if present
    logger.info(f"Unique coordinate IDs: {coordinates_df['teamId'].unique()}")

    # 3. Interpolate tracks
    detections_df = interpolate_tracks(detections_df, max_interpolation_gap)
    print(f"Number of unique tracklets: {detections_df['id'].nunique()}")

    # 4. Project all coordinates to image space once
    coord_points = np.array([[x * pitch_length, y * pitch_width] for x, y in coordinates_df[["x", "y"]].values])
    coord_points = coord_points.reshape(-1, 1, 2).astype(np.float32)
    projected_coords = cv2.perspectiveTransform(coord_points, H).reshape(-1, 2)

    # 5. Associate tracklets with coordinates and update detections
    detections_df = associate_tracklets_to_coordinates(
        detections_df,
        coordinates_df,
        projected_coords,
        max_distance=max_distance,
    )
    print(f"Number of unique tracklets: {detections_df['id'].nunique()}")

    # 6. Perform coordinate-aware interpolation
    # detections_df = interpolate_tracks_with_coordinates(
    #     detections_df, coordinates_df, projected_coords, max_gap=max_interpolation_gap
    # )

    # 7. Refine associations based on reprojection error
    # detections_df = iterative_association_refinement(
    #     detections_df,
    #     coordinates_df,
    #     projected_coords,
    #     max_iterations=max_refinement_iterations,
    #     error_threshold=error_threshold,
    # )

    # 8. Calculate final reprojection error
    # final_avg_error, final_track_errors = calculate_reprojection_error(
    #     detections_df, coordinates_df, projected_coords
    # )
    # logger.info(f"Final average reprojection error: {final_avg_error:.2f} pixels")

    # 9. Create final MOT format using updated detections_df
    print(detections_df.head())
    # mot_rows = []
    # for frame in sorted(detections_df["frame"].unique()):
    #     frame_dets = detections_df[detections_df["frame"] == frame]
    #     frame_coords = coordinates_df[coordinates_df["frame"] == frame]

    #     if frame_coords.empty or frame_dets.empty:
    #         continue

    #     for _, det in frame_dets.iterrows():
    #         coord_id = det["id"]
    #         if coord_id not in frame_coords["id"].values:
    #             continue  # Skip detections without a valid coordinate association

    #         # Find corresponding coordinate data
    #         coord_data = frame_coords[frame_coords["id"] == coord_id]
    #         if coord_data.empty:
    #             continue

    #         coord = coord_data.iloc[0]

    #         mot_row = {
    #             "frame": frame,
    #             "id": coord_id,  # Use coordinate ID for consistency
    #             "bb_left": det["bb_left"],
    #             "bb_top": det["bb_top"],
    #             "bb_width": det["bb_width"],
    #             "bb_height": det["bb_height"],
    #             "conf": det["conf"],
    #             "x": coord["x"],  # Use coordinate positions
    #             "y": coord["y"],
    #             "z": 0,  # Z coordinate is typically 0 in 2D tracking
    #         }
    #         mot_rows.append(mot_row)

    # Create and save MOT DataFrame
    mot_df = pd.DataFrame(detections_df)
    mot_df = mot_df.sort_values(["frame", "id"])

    logger.info(f"MOT DataFrame shape: {mot_df.shape}")
    logger.info(f"dataframe head \n{mot_df.head()}")
    mot_df.to_csv(output_path, index=False, header=False)
    logger.info(f"Ground truth MOT file saved to: {output_path}")


if __name__ == "__main__":
    # This allows the command to be run directly for testing
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Create ground truth MOT file.")
    parser.add_argument(
        "--config_path", type=str, default="configs/default_config.yaml", help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    create_ground_truth_mot(**config.create_ground_truth_mot)

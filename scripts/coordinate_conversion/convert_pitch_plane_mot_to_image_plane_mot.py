"""
This script converts raw XML tracking data to MOT-style CSV files using pitch plane coordinates.

Usage:
    python coordinate_conversion/convert_raw_to_pitch_plane_mot.py --match_id <match_id>

Example:
    python coordinate_conversion/convert_raw_to_pitch_plane_mot.py --match_id 117093

Arguments:
    --match_id: The match identifier to process.
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from loguru import logger


def redistort_points_fisheye(undistorted_pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Redistort undistorted image-plane points using fisheye model.

    Args:
        undistorted_pts: (N, 2) array of 'ideal' or 'undistorted' 2D points in pixel coordinates.
        K: (3, 3) camera intrinsic matrix.
        D: (4,) or (4, 1) fisheye distortion coefficients.

    Returns:
        (N, 2) array of distorted points (i.e., as they would appear in the real fisheye image).
    """
    # 1) Convert to normalized coords via K^-1
    K_inv = np.linalg.inv(K)

    undistorted_pts = undistorted_pts.reshape(-1, 2)
    num_points = undistorted_pts.shape[0]

    # Homogeneous pixel -> normalized coords
    normalized_3d = []
    for i in range(num_points):
        u, v = undistorted_pts[i]
        uv_hom = np.array([u, v, 1.0], dtype=np.float32)
        xyz = K_inv @ uv_hom
        # xyz is (x, y, z) in normalized coords; typically z ~ 1.0
        normalized_3d.append(xyz)

    normalized_3d = np.array(normalized_3d).reshape(-1, 1, 3)  # (N,1,3)

    # 2) Use fisheye.projectPoints to get lens-distorted pixel coords
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    distorted_pts_3d, _ = cv2.fisheye.projectPoints(normalized_3d, rvec, tvec, K, D)
    # shape (N,1,2) -> (N,2)
    distorted_pts = distorted_pts_3d.reshape(-1, 2)
    return distorted_pts


def parse_xml(xml_path):
    """
    Parse the XML file and extract tracking data.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        list of dict: A list containing tracking information for each player in each frame.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracking_data = []

    last_first_half_frame = 0
    current_period = "FIRST_HALF"
    frame_offset = 0

    for frame in root.findall("frame"):
        frame_number = int(frame.get("frameNumber"))
        event_period = frame.get("eventPeriod")

        if event_period != current_period:
            if current_period == "FIRST_HALF":
                last_first_half_frame = frame_number
                frame_offset = last_first_half_frame
            current_period = event_period

        if current_period == "SECOND_HALF":
            frame_number += frame_offset

        for player in frame.findall("player"):
            player_id = player.get("playerId")
            loc = player.get("loc")
            # Convert loc string to float coordinates
            try:
                x, y = map(float, loc.strip("[]").split(","))
                tracking_data.append({"frame": frame_number, "id": player_id, "x": x, "y": y})
            except ValueError:
                logger.warning(f"Invalid location format for player {player_id} in frame {frame_number}")

    return tracking_data


def write_csv(tracking_data, output_csv):
    """
    Write the tracking data to a CSV file in MOT-style format.

    Args:
        tracking_data (list of dict): Tracking information.
        output_csv (str): Path to the output CSV file.
    """
    fieldnames = ["frame", "id", "x", "y"]
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in tracking_data:
            writer.writerow(data)

    logger.info(f"Tracking data written to {output_csv}")


def convert_coordinates(
    coordinates_df: pd.DataFrame,
    homography: np.ndarray,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    calibrated: bool = True,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Convert coordinates from pitch plane to either calibrated image plane or distorted pitch plane coordinates.

    Args:
        coordinates_df: DataFrame containing pitch plane coordinates in [0..1] x [0..1].
        homography: 3x3 homography matrix for transforming pitch-plane -> image-plane.
        pitch_length: Length of the pitch in meters.
        pitch_width: Width of the pitch in meters.
        calibrated: Flag to toggle whether homography is applied.
                   - True  => Output the standard "image-plane" coords (homography applied).
                   - False => Output the "distorted" pitch-plane coords (no homography).
        K: Camera intrinsic matrix (3x3) for fisheye redistortion. Required if calibrated=False.
        D: Distortion coefficients (4,) for fisheye redistortion. Required if calibrated=False.

    Returns:
        DataFrame with either calibrated or "distorted" coordinates, depending on the flag.
    """
    # Copy input DataFrame so we don't overwrite
    output_df = coordinates_df.copy()

    # 1) Scale from normalized [0..1] into [0..pitch_length/width].
    pitch_coords = coordinates_df[["x", "y"]].values.astype(np.float32)
    pitch_coords[:, 0] *= pitch_length
    pitch_coords[:, 1] *= pitch_width

    # 2) Use homography to get *image-plane* (calibrated) coords
    pitch_coords_reshaped = pitch_coords.reshape(-1, 1, 2)
    image_coords = cv2.perspectiveTransform(pitch_coords_reshaped, homography)
    image_coords = image_coords.reshape(-1, 2)

    if calibrated:
        # 3a) If calibrated, use the homography result directly
        final_coords = image_coords
    else:
        # 3b) If not calibrated, apply fisheye redistortion
        if K is None or D is None:
            raise ValueError(
                "To output redistorted coordinates, you must provide camera intrinsics K and distortion D."
            )
        final_coords = redistort_points_fisheye(image_coords, K, D)

    output_df["x"] = final_coords[:, 0]
    output_df["y"] = final_coords[:, 1]

    return output_df


def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(description="Convert raw XML tracking data to MOT-style CSV using match_id.")
    parser.add_argument("--match_id", required=True, type=str, help="The match identifier to process.")
    parser.add_argument("--input_csv", required=True, type=str, help="Path to input CSV with pitch plane coordinates.")
    parser.add_argument("--homography_path", required=True, type=str, help="Path to homography matrix .npy file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save output files.")
    parser.add_argument(
        "--calibrated", action="store_true", default=True, help="Whether to output calibrated coordinates."
    )
    parser.add_argument(
        "--camera_intrinsics_path", type=str, help="Path to camera intrinsics .npy file (required if not calibrated)."
    )
    parser.add_argument("--pitch_length", type=float, default=105.0, help="Pitch length in meters.")
    parser.add_argument("--pitch_width", type=float, default=68.0, help="Pitch width in meters.")
    args = parser.parse_args()

    match_id = args.match_id
    input_csv = Path(args.input_csv)
    homography_path = Path(args.homography_path)
    output_dir = Path(args.output_dir)
    calibrated = args.calibrated
    camera_intrinsics_path = Path(args.camera_intrinsics_path) if args.camera_intrinsics_path else None

    if not input_csv.exists():
        logger.error(f"Input CSV file does not exist: {input_csv}")
        return

    if not homography_path.exists():
        logger.error(f"Homography matrix file does not exist: {homography_path}")
        return

    if not calibrated and (camera_intrinsics_path is None or not camera_intrinsics_path.exists()):
        logger.error("Camera intrinsics path is required when calibrated=False")
        return

    # Load data
    logger.info(f"Parsing CSV file: {input_csv}")
    coordinates_df = pd.read_csv(input_csv)

    logger.info(f"Loading homography matrix from {homography_path}")
    homography = np.load(homography_path)

    # Load camera intrinsics if needed
    K, D = None, None
    if not calibrated:
        logger.info(f"Loading camera intrinsics from {camera_intrinsics_path}")
        intrinsics_data = np.load(camera_intrinsics_path, allow_pickle=True)
        K = intrinsics_data["K"]
        D = intrinsics_data["D"]

    # Convert coordinates
    output_df = convert_coordinates(
        coordinates_df,
        homography=homography,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        calibrated=calibrated,
        K=K,
        D=D,
    )

    # Set up output path
    cal_suffix = "_calibrated" if calibrated else "_redistorted"
    output_csv = output_dir / f"{match_id}_image_plane_coordinates{cal_suffix}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    logger.info(f"Writing coordinates to {output_csv}")
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Successfully created coordinates CSV: {output_csv}")


if __name__ == "__main__":
    main()

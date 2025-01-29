"""Functions for converting pitch plane coordinates to image plane coordinates.

Usage:
    There are two ways to run the coordinate conversion:

    1. Using the Python module directly:
    ```bash
    python -m src.main command=convert_pitch_plane_to_image_plane \\
        convert_pitch_plane_to_image_plane.match_id=117093 \\
        convert_pitch_plane_to_image_plane.input_csv_path="data/interim/pitch_plane_coordinates/117093/117093_pitch_plane_coordinates.csv" \\
        convert_pitch_plane_to_image_plane.homography_path="data/interim/homography/117093/117093_homography.npy" \\
        convert_pitch_plane_to_image_plane.output_dir="data/interim/image_plane_coordinates/117093" \\
        convert_pitch_plane_to_image_plane.calibrated=true
    ```

    2. Using the convenience shell script (recommended):
    ```bash
    # First ensure the script is executable
    chmod +x scripts/convert_pitch_plane_to_image_plane.sh
    
    # Then run with match ID
    ./scripts/convert_pitch_plane_to_image_plane.sh 117093
    ```

Notes:
    - When calibrated=True (default), the code uses a homography to map from pitch-plane coordinates 
      to image-plane coordinates. This does not model lens distortion directly (like fish-eye or 
      radial distortion).
    - When calibrated=False, the code outputs "distorted pitch-plane coordinates," which are just 
      the raw pitch coordinates scaled by the pitch dimensions (meters) without applying any homography.
    - For true "lens re-distortion," you would need camera intrinsics and distortion coefficients 
      and to use OpenCV's cv2.projectPoints or cv2.fisheye.projectPoints. This is a more advanced 
      use case not covered by this script.
"""

import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List
import pandas as pd

from loguru import logger


def load_coordinates(coordinates_path: Path | str, match_id: str, event_period: str | None = None) -> pd.DataFrame:
    """
    Load coordinates from CSV file and optionally filter by event period.

    Args:
        coordinates_path: Path to the coordinates CSV file.
        match_id: Match ID for filtering.
        event_period: If provided, filter coordinates by event period.

    Returns:
        DataFrame containing the coordinates.
    """
    df = pd.read_csv(coordinates_path)
    if event_period:
        df = df[df["event_period"] == event_period]
    return df


def redistort_points_fisheye(
    undistorted_pts: np.ndarray, new_K: np.ndarray, K: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """
    Convert 'undistorted' points (in new_K's pixel coords) back to the original
    fisheye-distorted coordinate system described by (K, D).

    Args:
        undistorted_pts: (N, 2) array of points in the 'undistorted' pixel coordinate system,
                         i.e., what you get from cv2.fisheye.undistortPoints(..., P=new_K).
        new_K: (3, 3) the camera matrix used for undistortion (estimateNewCameraMatrixForUndistortRectify).
        K: (3, 3) the ORIGINAL camera intrinsic matrix.
        D: (4,) or (4,1) fisheye distortion coefficients for the ORIGINAL camera.

    Returns:
        (N, 2) array of points in the original fisheye-distorted image pixel coords.
    """
    # 1) Convert from the 'new_K' pixel coords -> normalized rays
    new_K_inv = np.linalg.inv(new_K)

    undistorted_pts = undistorted_pts.reshape(-1, 2)
    num_points = undistorted_pts.shape[0]

    # Build homogeneous coords & apply new_K^-1
    normalized_3d = []
    for i in range(num_points):
        u, v = undistorted_pts[i]
        uv_hom = np.array([u, v, 1.0], dtype=np.float32)
        xyz = new_K_inv @ uv_hom  # normalized camera coordinates
        normalized_3d.append(xyz)

    normalized_3d = np.array(normalized_3d, dtype=np.float32).reshape(-1, 1, 3)

    # 2) Use the ORIGINAL camera model (K, D) to re-distort
    rvec = np.zeros((3, 1), dtype=np.float32)  # no extra rotation
    tvec = np.zeros((3, 1), dtype=np.float32)  # no extra translation

    distorted_pts_3d, _ = cv2.fisheye.projectPoints(normalized_3d, rvec, tvec, K, D)
    distorted_pts = distorted_pts_3d.reshape(-1, 2)

    return distorted_pts


def convert_coordinates(
    coordinates_df: pd.DataFrame,
    homography: np.ndarray,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    calibrated: bool = True,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
    new_K: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Convert pitch-plane coordinates to the image plane, then either:
      - (calibrated=True) keep them 'undistorted' (homography result).
      - (calibrated=False) apply lens re-distortion to the homography result,
        using the fisheye model with camera intrinsics K, D.

    Args:
        coordinates_df: DataFrame with columns ["x", "y"] in normalized pitch coords [0..1].
        homography: 3x3 homography (planar) for pitch->image transform.
        pitch_length: Real pitch length in meters.
        pitch_width: Real pitch width in meters.
        calibrated: Flag controlling whether to re-distort or not.
        K: (3, 3) camera intrinsic matrix (if redistorting).
        D: (4,) or (4,1) fisheye distortion coefficients (if redistorting).

    Returns:
        DataFrame with either undistorted or redistorted image-plane coords in columns ["x","y"].
    """
    output_df = coordinates_df.copy()

    # 1) Scale from normalized [0..1] to real meters
    pitch_coords = coordinates_df[["x", "y"]].values.astype(np.float32)
    pitch_coords[:, 0] *= pitch_length
    pitch_coords[:, 1] *= pitch_width

    # 2) Homography -> 'ideal' or 'undistorted' image-plane coords
    pitch_coords_reshaped = pitch_coords.reshape(-1, 1, 2)  # (N,1,2)
    undistorted_coords = cv2.perspectiveTransform(pitch_coords_reshaped, homography)
    undistorted_coords = undistorted_coords.reshape(-1, 2)  # (N,2)

    # 3) If user wants 'calibrated' => return homography result
    #    If user wants 'un-calibrated' => redistort using fisheye model
    if calibrated:
        final_coords = undistorted_coords
    else:
        if K is None or D is None or new_K is None:
            raise ValueError(
                "To output redistorted coordinates, you must provide camera intrinsics K and distortion D."
            )
        final_coords = redistort_points_fisheye(undistorted_coords, new_K, K, D)

    output_df["x"] = final_coords[:, 0]
    output_df["y"] = final_coords[:, 1]
    return output_df


def convert_pitch_plane_to_image_plane(
    match_id: str,
    input_csv_path: Path | str,
    homography_path: Path | str,
    output_dir: Path | str,
    event_period: str | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    calibrated: bool = True,
    camera_intrinsics_path: Path | str | None = None,
) -> Path:
    """
    Top-level function to convert pitch-plane coords to image-plane coords,
    optionally re-distorting if calibrated=False.

    Args:
        match_id: Match ID for naming output files.
        input_csv_path: CSV file with ["x","y"] pitch coords in [0..1].
        homography_path: Numpy file .npy storing the 3x3 homography.
        output_dir: Where to save results.
        event_period: Optional filter for the CSV.
        pitch_length: Real pitch length in meters.
        pitch_width: Real pitch width in meters.
        calibrated: If True, output homography-based 'undistorted' coords.
                    If False, re-distort them with lens parameters.
        camera_intrinsics_path: .npy storing [K, D] for fisheye lens.
                                Required if calibrated=False.

    Returns:
        Path to the resulting CSV.
    """
    try:
        input_csv_path = Path(input_csv_path)
        homography_path = Path(homography_path)
        output_dir = Path(output_dir)

        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
        if not homography_path.exists():
            raise FileNotFoundError(f"Homography matrix file not found: {homography_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading coordinates from {input_csv_path}")
        coordinates_df = load_coordinates(input_csv_path, match_id, event_period)

        logger.info(f"Loading homography matrix from {homography_path}")
        homography = np.load(homography_path)

        # If we want redistorted coords, we must load intrinsics K, D
        K, D, new_K = None, None, None
        if not calibrated:
            if not camera_intrinsics_path:
                raise ValueError("calibrated=False but no camera_intrinsics_path provided!")
            camera_intrinsics_path = Path(camera_intrinsics_path)
            if not camera_intrinsics_path.exists():
                raise FileNotFoundError(f"Camera intrinsics file not found: {camera_intrinsics_path}")

            logger.info(f"Loading camera intrinsics (K, D) from {camera_intrinsics_path}")
            intrinsics_data = np.load(camera_intrinsics_path, allow_pickle=True)
            # Expect intrinsics_data to be a dict or a tuple [K, D]
            # Adjust to match however you saved them. For example:
            # intrinsics_data = {'K': <3x3>, 'D': <4x1>}
            K = intrinsics_data["K"]
            D = intrinsics_data["D"]
            new_K = intrinsics_data["Knew"]
        # Convert
        logger.info(
            "Applying homography and then {}distorting with lens parameters".format("NOT " if calibrated else "")
        )
        output_df = convert_coordinates(
            coordinates_df,
            homography=homography,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            calibrated=calibrated,
            K=K,
            D=D,
            new_K=new_K,
        )

        period_map = {"FIRST_HALF": "1st_half", "SECOND_HALF": "2nd_half"}
        period_suffix = f"_{period_map[event_period]}" if event_period else ""
        cal_suffix = "_calibrated" if calibrated else "_distorted"
        output_csv = output_dir / f"{match_id}_image_plane_coordinates{period_suffix}{cal_suffix}.csv"

        logger.info(f"Writing output CSV to {output_csv}")
        output_df.to_csv(output_csv, index=False)

        logger.info(f"Successfully created coordinates CSV: {output_csv}")
        return output_csv

    except (FileNotFoundError, Exception) as e:
        logger.error(str(e))
        raise

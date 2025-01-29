"""Module for generating camera calibration mappings from keypoint annotations."""

import json
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
from loguru import logger


def load_keypoints(keypoints_path: Path | str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load keypoints from a JSON file and convert to object and image points.

    Args:
        keypoints_path: Path to the JSON file containing keypoints

    Returns:
        Tuple containing:
        - Object points (Nx1x3 array)
        - Image points (Nx1x2 array)
        - Original keypoints dictionary
    """
    logger.info(f"Loading keypoints from JSON file: {keypoints_path}")
    with open(keypoints_path, "r") as f:
        image_points_dict = json.load(f)

    world_points_dict = {}
    for key in image_points_dict.keys():
        coord = tuple(map(float, key.strip("()").split(",")))
        world_points_dict[key] = [coord[0], coord[1], 0.0]  # Z-coordinate is 0

    logger.info("Converting image and object points to numpy arrays")
    imgpoints = np.array(list(image_points_dict.values()), dtype=np.float32)
    objpoints = np.array(list(world_points_dict.values()), dtype=np.float32)

    imgpoints = imgpoints.reshape(-1, 1, 2)
    objpoints = objpoints.reshape(-1, 1, 3)
    return objpoints, imgpoints, image_points_dict


def calibrate_keypoints(imgpoints: np.ndarray, K: np.ndarray, D: np.ndarray, Knew: np.ndarray) -> np.ndarray:
    """
    Calibrate keypoints using camera parameters.

    Args:
        imgpoints: Image points (Nx1x2 array)
        K: Camera matrix
        D: Distortion coefficients
        Knew: New camera matrix

    Returns:
        Calibrated points (Nx2 array)
    """
    logger.info("Calibrating keypoints")
    undistorted_points = cv2.fisheye.undistortPoints(imgpoints, K, D, P=Knew)
    undistorted_points = undistorted_points.reshape(-1, 2)
    return undistorted_points


def generate_calibration_mappings(
    match_id: str,
    keypoints_path: Path | str,
    video_path: Path | str,
    output_dir: Path | str,
) -> None:
    """
    Generate calibration mappings from keypoints and save them.

    Args:
        match_id: The ID of the match
        keypoints_path: Path to the keypoints JSON file
        video_path: Path to the video file (used for dimensions)
        output_dir: Directory to save the calibration files
    """
    # Convert paths to Path objects
    keypoints_path = Path(keypoints_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check input files exist
    if not keypoints_path.exists() or not video_path.exists():
        logger.error("Input files missing")
        return

    # Load keypoints
    objpoints, imgpoints, original_keypoints = load_keypoints(keypoints_path)

    # Get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error(f"Failed to read from video file: {video_path}")
        return
    image_height, image_width, _ = frame.shape

    # Camera calibration
    logger.info("Starting camera calibration")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_K3
        + cv2.fisheye.CALIB_FIX_K4
    )
    criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)

    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        [objpoints],
        [imgpoints],
        (image_width, image_height),
        K,
        D,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )
    logger.info(f"Calibration RMS error: {retval}")

    # Generate undistortion maps
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (image_width, image_height), np.eye(3), balance=1
    )
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (image_width, image_height), cv2.CV_16SC2)

    # Calibrate keypoints
    calibrated_keypoints = calibrate_keypoints(imgpoints, K, D, new_K)

    # Save mappings
    np.save(output_dir / f"{match_id}_mapx.npy", mapx)
    np.save(output_dir / f"{match_id}_mapy.npy", mapy)

    # Save calibrated keypoints
    calibrated_keypoints_dict = {}
    for (key, _), calibrated_point in zip(original_keypoints.items(), calibrated_keypoints):
        calibrated_keypoints_dict[key] = calibrated_point.tolist()

    with open(output_dir / f"{match_id}_calibrated_keypoints.json", "w") as f:
        json.dump(calibrated_keypoints_dict, f, indent=2)

    logger.info(f"Saved mapx, mapy, and calibrated keypoints to {output_dir}")

    # SAVE CAMERA INTRINSICS
    # We include K, D, new_K, plus optional rvecs/tvecs and the RMS (retval).
    # We'll store them all in an .npz file for easy reloading.
    np.savez(
        output_dir / f"{match_id}_camera_intrinsics.npz",
        K=K,
        D=D,
        Knew=new_K,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        rms=retval,
    )
    logger.info(f"Saved camera intrinsics to {output_dir / f'{match_id}_camera_intrinsics.npz'}")


def main():
    """Command line interface for generating calibration mappings."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate camera calibration mappings from keypoints")
    parser.add_argument("--match_id", required=True, help="The ID of the match")
    parser.add_argument("--keypoints_path", type=str, help="Path to keypoints JSON file")
    parser.add_argument("--video_path", type=str, help="Path to video file")
    parser.add_argument("--output_dir", type=str, help="Directory to save calibration files")
    args = parser.parse_args()

    # Set up default paths if not provided
    base_path = Path("/data/share/SoccerTrack-v2/data")
    keypoints_path = (
        Path(args.keypoints_path)
        if args.keypoints_path
        else base_path / "raw" / args.match_id / f"{args.match_id}_keypoints.json"
    )
    video_path = (
        Path(args.video_path)
        if args.video_path
        else base_path / "raw" / args.match_id / f"{args.match_id}_panorama.mp4"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir else base_path / "interim" / "calibrated_keypoints" / args.match_id
    )

    generate_calibration_mappings(
        match_id=args.match_id,
        keypoints_path=keypoints_path,
        video_path=video_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

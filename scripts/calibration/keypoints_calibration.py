"""
This script calibrates a camera using keypoints from a JSON file and a checkerboard pattern video.
It also calibrates the keypoints and saves them in a specified directory.

Usage:
    python keypoints_calibration.py --match_id <match_id>

Arguments:
    --match_id: The ID of the match.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from loguru import logger
import cv2


def load_json(keypoints_json_path):
    logger.info(f"Loading keypoints from JSON file: {keypoints_json_path}")
    with open(keypoints_json_path, "r") as f:
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


def calibrate_keypoints(imgpoints, K, D, Knew):
    logger.info("Calibrating keypoints")
    undistorted_points = cv2.fisheye.undistortPoints(imgpoints, K, D, P=Knew)
    undistorted_points = undistorted_points.reshape(-1, 2)
    return undistorted_points


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera and keypoints.")
    parser.add_argument("--match_id", required=True, help="The ID of the match")
    args = parser.parse_args()

    match_folder = Path(f"data/raw/{args.match_id}")
    keypoints_json_path = match_folder / f"{args.match_id}_keypoints.json"
    video_path = match_folder / f"{args.match_id}_panorama.mp4"

    if not keypoints_json_path.exists():
        logger.error(f"Keypoints file not found: {keypoints_json_path}")
        return
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    logger.info("Loading object points and image points")
    objpoints, imgpoints, original_keypoints = load_json(keypoints_json_path)

    logger.info("Retrieving video dimensions")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error(f"Failed to read from video file: {video_path}")
        return
    image_height, image_width, _ = frame.shape

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
        [objpoints], [imgpoints], (image_width, image_height), K, D, None, None, flags=flags, criteria=criteria
    )
    logger.info(f"Calibration RMS error: {retval}")

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (image_width, image_height), alpha=0.98, centerPrincipalPoint=1)
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (image_width, image_height), cv2.CV_16SC2)

    logger.info("Calibrating keypoints")
    calibrated_keypoints = calibrate_keypoints(imgpoints, K, D, new_K)

    # Save the mappings to files
    save_path = Path("/home/atom/SoccerTrack-v2/data/interim/calibrated_keypoints")
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / f"{args.match_id}_mapx.npy", mapx)
    np.save(save_path / f"{args.match_id}_mapy.npy", mapy)

    # Save calibrated keypoints
    calibrated_keypoints_dict = {}
    for (key, _), calibrated_point in zip(original_keypoints.items(), calibrated_keypoints):
        calibrated_keypoints_dict[key] = calibrated_point.tolist()

    with open(save_path / f"{args.match_id}_calibrated_keypoints.json", "w") as f:
        json.dump(calibrated_keypoints_dict, f, indent=2)

    logger.info(f"Saved mapx, mapy, and calibrated keypoints to {save_path}")


if __name__ == "__main__":
    main()


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Calibrate camera using a checkerboard pattern.")

    # Add the arguments
    parser.add_argument(
        "--keypoints_json_path", type=str, required=True, help="The path to the JSON file containing keypoints"
    )
    parser.add_argument("--input_mp4_path", type=str, required=True, help="The path to the input mp4 file")

    # Parse the arguments
    args = parser.parse_args()
    keypoints_json_path = Path(args.keypoints_json_path)

    logger.info("Loading object points and image points")
    objpoints, imgpoints = load_json(keypoints_json_path)
    logger.info("Retrieving video dimensions")

    cap = cv2.VideoCapture(args.input_mp4_path)
    ret, frame = cap.read()
    image_height, image_width, _ = frame.shape
    dim = (image_width, image_height)

    logger.info("Starting camera calibration")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, (image_width, image_height), K, D, flags=flags
    )

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (image_width, image_height), alpha=0.98, centerPrincipalPoint=1)

    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (image_width, image_height), cv2.CV_16SC2)

    logger.info(f"Calibration complete.")

    # Save the mappings to files
    save_path = keypoints_json_path.parent
    np.save(save_path / "mapx.npy", mapx)
    np.save(save_path / "mapy.npy", mapy)

    logger.info(f"Saved mapx and mapy to {save_path}")


if __name__ == "__main__":
    main()

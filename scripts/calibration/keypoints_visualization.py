"""
This script visualizes original and calibrated keypoints on the first frame of a video for a given match ID.

Usage:
    python keypoints_visualization.py --match_id <match_id>

Arguments:
    --match_id: The ID of the match.

Output:
    The script saves two images in the 'data/interim/keypoints_visualization/<match_id>/' directory:
    - <match_id>_original_keypoints.jpg: Original frame with keypoints
    - <match_id>_calibrated_keypoints.jpg: Calibrated frame with keypoints
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from loguru import logger


def load_json(keypoints_json_path):
    logger.info(f"Loading keypoints from JSON file: {keypoints_json_path}")
    with open(keypoints_json_path, "r") as f:
        image_points_dict = json.load(f)
    imgpoints = np.array(list(image_points_dict.values()), dtype=np.float32)
    imgpoints = imgpoints.reshape(-1, 1, 2)
    return imgpoints


def plot_keypoints(img, keypoints, color=(0, 0, 255)):
    for point in np.squeeze(keypoints):
        x, y = point.ravel()
        logger.info(f"keypoint: {x=}, {y=}")
        cv2.circle(img, (int(x), int(y)), 10, color, -1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Keypoints Visualization")
    parser.add_argument("--match_id", required=True, help="The ID of the match")
    args = parser.parse_args()

    # Construct the paths
    match_folder = Path(f"data/raw/{args.match_id}")
    interim_folder = Path("data/interim")
    keypoints_json_path = match_folder / f"{args.match_id}_keypoints.json"
    video_path = match_folder / f"{args.match_id}_panorama.mp4"
    calibrated_video_frame_path = (
        interim_folder / "calibrated_videos" / f"{args.match_id}" / f"{args.match_id}_panorama.jpg"
    )
    calibrated_keypoints_path = (
        interim_folder / "calibrated_keypoints" / f"{args.match_id}" / f"{args.match_id}_calibrated_keypoints.json"
    )

    # Set output folder
    output_folder = interim_folder / "keypoints_visualization" / args.match_id
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load keypoints
    original_keypoints = load_json(str(keypoints_json_path))
    calibrated_keypoints = load_json(str(calibrated_keypoints_path))

    # Read original and calibrated frames
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error(f"Failed to read the first frame from {video_path}")
        return

    calibrated_frame = cv2.imread(str(calibrated_video_frame_path))

    # Plot keypoints on the frames
    original_frame_with_keypoints = plot_keypoints(frame.copy(), original_keypoints, color=(0, 0, 255))
    calibrated_frame_with_keypoints = plot_keypoints(calibrated_frame.copy(), calibrated_keypoints, color=(0, 255, 0))

    # Save the resulting images
    cv2.imwrite(str(output_folder / f"{args.match_id}_original_keypoints.jpg"), original_frame_with_keypoints)
    cv2.imwrite(str(output_folder / f"{args.match_id}_calibrated_keypoints.jpg"), calibrated_frame_with_keypoints)
    logger.info(f"Images with keypoints saved to {output_folder}")


if __name__ == "__main__":
    main()

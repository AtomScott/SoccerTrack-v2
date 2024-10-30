"""
This script calculates the homography matrix using keypoints from a JSON file and saves it to an interim folder.

Usage:
    python calculate_homography.py --match_id <match_id>

Arguments:
    --match_id: The ID of the match.
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from loguru import logger


def load_keypoints(keypoints_json_path):
    logger.info(f"Loading keypoints from JSON file: {keypoints_json_path}")
    with open(keypoints_json_path, "r") as f:
        keypoints_dict = json.load(f)

    image_points = []
    world_points = []
    for key, value in keypoints_dict.items():
        world_coord = tuple(map(float, key.strip("()").split(",")))
        image_coord = value
        world_points.append(world_coord)
        image_points.append(image_coord)

    return np.array(world_points, dtype=np.float32), np.array(image_points, dtype=np.float32)


def calculate_homography(world_points, image_points):
    logger.info("Calculating homography matrix")
    H, _ = cv2.findHomography(world_points[:, :2], image_points, cv2.RANSAC, 5.0)
    return H


def main():
    parser = argparse.ArgumentParser(description="Calculate homography matrix")
    parser.add_argument("--match_id", required=True, help="The ID of the match")
    args = parser.parse_args()

    # Construct the paths
    match_folder = Path(f"data/interim/calibrated_keypoints/{args.match_id}")
    keypoints_json_path = match_folder / f"{args.match_id}_calibrated_keypoints.json"

    # Check if file exists
    if not keypoints_json_path.exists():
        logger.error(f"Keypoints file not found: {keypoints_json_path}")
        return

    # Load keypoints
    world_points, image_points = load_keypoints(str(keypoints_json_path))

    # Calculate homography
    H = calculate_homography(world_points, image_points)

    # Create interim folder if it doesn't exist
    interim_folder = Path("data/interim") / "homography" / f"{args.match_id}"
    interim_folder.mkdir(parents=True, exist_ok=True)

    # Save homography matrix
    homography_path = interim_folder / f"{args.match_id}_homography.npy"
    np.save(homography_path, H)
    logger.info(f"Homography matrix saved to {homography_path}")


if __name__ == "__main__":
    main()

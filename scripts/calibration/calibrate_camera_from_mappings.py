"""
This script calibrates videos using precomputed mapping files (mapx.npy and mapy.npy).

Usage:
    python calibrate_camera_from_mappings.py --match_id <match_id> [--n_jobs <n_jobs>] [--overwrite] [--first_frame_only]

Arguments:
    --match_id: The ID of the match. This is used to locate the input files and name the output files.
    --n_jobs: Number of parallel jobs (default: 1).
    --overwrite: Overwrite existing files in the output folder.
    --first_frame_only: Calibrate only the first frame of each video.

Input files (expected in /home/atom/SoccerTrack-v2/data/interim/calibrated_keypoints/<match_id>/):
    - <match_id>_mapx.npy: X-axis mapping for undistortion
    - <match_id>_mapy.npy: Y-axis mapping for undistortion

Input video (expected in /home/atom/SoccerTrack-v2/data/raw/<match_id>/):
    - <match_id>_panorama.mp4: Panorama video file

Output file (saved in /home/atom/SoccerTrack-v2/data/interim/calibrated_videos/<match_id>/):
    - <match_id>_panorama.mp4: Calibrated video file (or .jpg if first_frame_only is used)
"""

import cv2
import numpy as np
import argparse
from loguru import logger
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from deffcode import FFdecoder
from vidgear.gears import WriteGear
import json


def get_fps(video_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata(video_path)
        # Attempt to retrieve FPS from different possible keys
        fps = metadata[0].get("Video", {}).get("FrameRate") or metadata[0].get("Video", {}).get("VideoFrameRate")

        if fps is None:
            logger.warning(f"FPS not found in metadata for video: {video_path}. Trying OpenCV as fallback.")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            else:
                logger.error(f"Failed to open video file: {video_path}")
                return None  # or set a default value

    return fps


def calibrate_video(match_id, overwrite, first_frame_only):
    input_folder = Path(f"/home/atom/SoccerTrack-v2/data/raw/{match_id}")
    map_folder = Path(f"/home/atom/SoccerTrack-v2/data/interim/calibrated_keypoints/{match_id}")
    output_folder = Path(f"/home/atom/SoccerTrack-v2/data/interim/calibrated_videos/{match_id}")
    output_folder.mkdir(parents=True, exist_ok=True)

    video_path = input_folder / f"{match_id}_panorama_1st_half.mp4"
    mapx_path = map_folder / f"{match_id}_mapx.npy"
    mapy_path = map_folder / f"{match_id}_mapy.npy"
    save_path = output_folder / (f"{match_id}_panorama.jpg" if first_frame_only else f"{match_id}_panorama.mp4")

    if not overwrite and save_path.exists():
        logger.info(f"Skipping existing file {save_path}")
        return

    if not video_path.exists() or not mapx_path.exists() or not mapy_path.exists():
        logger.warning(f"Missing input files for match_id {match_id}")
        return

    # Load mapx and mapy
    mapx = np.load(mapx_path)
    mapy = np.load(mapy_path)

    # Initialize the decoder
    decoder = FFdecoder(str(video_path), frame_format="bgr24").formulate()

    if first_frame_only:
        # Process only the first frame
        frame = next(decoder.generateFrame())
        if frame is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(save_path), frame)
        decoder.terminate()
    else:
        # Process the entire video
        output_params = {
            "-input_framerate": json.loads(decoder.metadata)["output_framerate"],
            "-vcodec": "h264_nvenc",
        }
        writer = WriteGear(output=str(save_path), logging=True, **output_params)

        for frame in tqdm(decoder.generateFrame(), desc="Processing frames"):
            if frame is None:
                break
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            writer.write(frame)
        writer.close()
        decoder.terminate()

    logger.info(f"Saved calibrated {'frame' if first_frame_only else 'video'} to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate video using precomputed mappings")
    parser.add_argument("--match_id", required=True, type=str, help="The ID of the match")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in the output folder")
    parser.add_argument("--first_frame_only", action="store_true", help="Calibrate only the first frame of the video")
    args = parser.parse_args()

    calibrate_video(args.match_id, args.overwrite, args.first_frame_only)


if __name__ == "__main__":
    main()

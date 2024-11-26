"""
This script calibrates videos using precomputed mapping files (mapx.npy and mapy.npy).

Usage:
    python calibrate_camera_from_mappings.py --match_id <match_id> [--custom_video <custom_video>] [--n_jobs <n_jobs>] [--overwrite] [--first_frame_only]

Arguments:
    --match_id: The ID of the match. This is used to locate the input files and name the output files.
    --custom_video: Path to custom video file to calibrate.
    --n_jobs: Number of parallel jobs (default: 1).
    --overwrite: Overwrite existing files in the output folder.
    --first_frame_only: Calibrate only the first frame of each video.

Input files (expected in /home/atom/SoccerTrack-v2/data/interim/calibrated_keypoints/<match_id>/):
    - <match_id>_mapx.npy: X-axis mapping for undistortion
    - <match_id>_mapy.npy: Y-axis mapping for undistortion

Input video:
    If --custom_video is not provided (expected in /home/atom/SoccerTrack-v2/data/raw/<match_id>/):
        - <match_id>_panorama_1st_half.mp4: Panorama video file
    If --custom_video is provided:
        - Path to any MP4 file you want to calibrate

Output file (saved in /home/atom/SoccerTrack-v2/data/interim/calibrated_videos/<match_id>/):
    If --custom_video is not provided:
        - <match_id>_panorama.mp4: Calibrated video file (or .jpg if first_frame_only is used)
    If --custom_video is provided:
        - <video_name>_calibrated.mp4: Calibrated video file (or .jpg if first_frame_only is used)
"""

import argparse
import json
from pathlib import Path
from exiftool import ExifToolHelper

import cv2
import numpy as np
from deffcode import FFdecoder
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from vidgear.gears import WriteGear

# Use the actual path instead of the symlink
BASE_PATH = Path("/data/share/SoccerTrack-v2/data")


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


def calibrate_video(match_id, overwrite, first_frame_only, custom_video=None):
    input_folder = BASE_PATH / "raw" / match_id
    map_folder = BASE_PATH / "interim" / "calibrated_keypoints" / match_id
    output_folder = BASE_PATH / "interim" / "calibrated_videos" / match_id

    # Create directories with explicit permissions
    for folder in [input_folder, map_folder, output_folder]:
        if not folder.exists():
            logger.warning(f"Folder {folder} does not exist. Creating...")
            folder.mkdir(parents=True, exist_ok=True)
            # Ensure group permissions are set correctly
            folder.chmod(0o2775)  # This sets rwxrwsr-x

    # Use custom video path if provided, otherwise use default path
    video_path = Path(custom_video) if custom_video else input_folder / f"{match_id}_panorama_1st_half.mp4"
    mapx_path = map_folder / f"{match_id}_mapx.npy"
    mapy_path = map_folder / f"{match_id}_mapy.npy"

    # Modify save path to use custom video name if provided
    if custom_video:
        video_stem = Path(custom_video).stem
        save_path = output_folder / (
            f"{video_stem}_calibrated.jpg" if first_frame_only else f"{video_stem}_calibrated.mp4"
        )
    else:
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
            "-vcodec": "h264",
        }
        # Add file:// scheme to the path
        output_path = f"file://{save_path.resolve()}"
        writer = WriteGear(output=output_path, logging=True, **output_params)

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
    parser.add_argument("--custom_video", type=str, help="Path to custom video file to calibrate")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in the output folder")
    parser.add_argument("--first_frame_only", action="store_true", help="Calibrate only the first frame of the video")
    args = parser.parse_args()

    calibrate_video(args.match_id, args.overwrite, args.first_frame_only, args.custom_video)


if __name__ == "__main__":
    main()

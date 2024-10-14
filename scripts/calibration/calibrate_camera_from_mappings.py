"""
This script calibrates videos using precomputed mapping files (mapx.npy and mapy.npy).

Usage:
    python calibrate_camera_from_mappings.py --input_folder <input_folder> --output_folder <output_folder> [--n_jobs <n_jobs>] [--overwrite]

Arguments:
    --input_folder: Folder containing video and map files.
    --output_folder: Folder to save calibrated videos.
    --n_jobs: Number of parallel jobs (default: 1).
    --overwrite: Overwrite existing files in the output folder.
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
from exiftool import ExifToolHelper
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


def calibrate_video(video_path, mapx_path, mapy_path, save_path, overwrite):
    if not overwrite and Path(save_path).exists():
        logger.info(f"Skipping existing file {save_path}")
        return

    # Load mapx and mapy
    mapx = np.load(mapx_path)
    mapy = np.load(mapy_path)

    # Vidgear Score
    # initialize and formulate the decoder
    decoder = FFdecoder(video_path, frame_format="bgr24").formulate()

    output_params = {
        "-input_framerate": json.loads(decoder.metadata)["output_framerate"],
        "-vcodec": "h264_nvenc",
    }

    writer = WriteGear(output=save_path, logging=True, **output_params)

    # Get total frames for tqdm
    for frame in tqdm(decoder.generateFrame(), desc="Processing frames"):
        if frame is None:
            break
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        writer.write(frame)
    writer.close()
    decoder.terminate()


def calibrate_videos_in_folder(input_folder, output_folder, n_jobs, overwrite):
    # Convert string paths to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Prepare list of tasks
    tasks = []
    for video_path in input_folder.glob("*.mp4"):
        mapx_path = input_folder / "mapx.npy"
        mapy_path = input_folder / "mapy.npy"

        logger.info(f"Processing video {video_path.name}")
        if not mapx_path.exists() or not mapy_path.exists():
            logger.warning(f"Missing map files for video {video_path.name}")
            continue

        save_path = output_folder / video_path.name
        tasks.append((str(video_path), str(mapx_path), str(mapy_path), str(save_path), overwrite))

    # Process videos in parallel
    Parallel(n_jobs=n_jobs)(delayed(calibrate_video)(*task) for task in tasks)


def main():
    parser = argparse.ArgumentParser(description="Batch Calibrate Videos")
    parser.add_argument(
        "--input_folder",
        required=True,
        type=str,
        help="Folder containing video and map files",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        type=str,
        help="Folder to save calibrated videos",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output folder",
    )
    args = parser.parse_args()

    calibrate_videos_in_folder(args.input_folder, args.output_folder, args.n_jobs, args.overwrite)


if __name__ == "__main__":
    main()

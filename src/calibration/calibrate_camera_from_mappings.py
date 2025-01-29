"""Module for calibrating videos using precomputed mapping files.

Usage:
    You can use this module in two ways:

    1. Calibrate a full video:
    ```bash
    python -m src.calibration.calibrate_camera_from_mappings \
        --match_id 117093 \
        --input_video "data/raw/117093/117093_panorama_1st_half.mp4" \
        --output_path "data/interim/calibrated_videos/117093/117093_panorama_1st_half.mp4"
    ```

    2. Calibrate just the first frame (useful for testing):
    ```bash
    python -m src.calibration.calibrate_camera_from_mappings \
        --match_id 117093 \
        --first_frame_only
    ```

    Required Files:
    - Input video: data/raw/<match_id>/<match_id>_panorama_1st_half.mp4
    - Mapping files (automatically located):
      - X-axis mapping: data/interim/calibrated_keypoints/<match_id>/<match_id>_mapx.npy
      - Y-axis mapping: data/interim/calibrated_keypoints/<match_id>/<match_id>_mapy.npy

    Output Files:
    - For full video: data/interim/calibrated_videos/<match_id>/<match_id>_panorama.mp4
    - For single frame: data/interim/calibrated_videos/<match_id>/<match_id>_panorama.jpg
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from deffcode import FFdecoder
from exiftool import ExifToolHelper
from loguru import logger
from tqdm import tqdm
from vidgear.gears import WriteGear


def get_fps(video_path: Path | str) -> float | None:
    """
    Get the FPS of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        float | None: The FPS of the video, or None if it couldn't be determined
    """
    with ExifToolHelper() as et:
        metadata = et.get_metadata(str(video_path))
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
                return None

    return fps


def calibrate_video(
    match_id: str,
    input_video_path: Path | str,
    mapx_path: Path | str,
    mapy_path: Path | str,
    output_path: Path | str,
    first_frame_only: bool = False,
) -> None:
    """
    Calibrate a video using precomputed mapping files.

    Args:
        match_id: The ID of the match
        input_video_path: Path to the input video file
        mapx_path: Path to the X-axis mapping file
        mapy_path: Path to the Y-axis mapping file
        output_path: Path where the calibrated video/frame will be saved
        first_frame_only: If True, only calibrate the first frame
    """
    # Convert paths to Path objects
    input_video_path = Path(input_video_path)
    mapx_path = Path(mapx_path)
    mapy_path = Path(mapy_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check input files exist
    if not all(p.exists() for p in [input_video_path, mapx_path, mapy_path]):
        logger.error("One or more input files missing")
        return

    # Load mapx and mapy
    mapx = np.load(mapx_path)
    mapy = np.load(mapy_path)

    # Initialize the decoder
    decoder = FFdecoder(str(input_video_path), frame_format="bgr24").formulate()

    if first_frame_only:
        # Process only the first frame
        frame = next(decoder.generateFrame())
        if frame is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(output_path), frame)
        decoder.terminate()
    else:
        # Process the entire video
        output_params = {
            "-input_framerate": get_fps(input_video_path),
            "-vcodec": "h264",
        }
        writer = WriteGear(output=f"file://{output_path.resolve()}", logging=True, **output_params)

        for frame in tqdm(decoder.generateFrame(), desc="Processing frames"):
            if frame is None:
                break
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            writer.write(frame)
        writer.close()
        decoder.terminate()

    logger.info(f"Saved calibrated {'frame' if first_frame_only else 'video'} to {output_path}")


def main():
    """Command line interface for video calibration."""
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate video using precomputed mappings")
    parser.add_argument("--match_id", required=True, type=str, help="The ID of the match")
    parser.add_argument("--input_video", type=str, help="Path to input video file")
    parser.add_argument("--output_path", type=str, help="Path to save the calibrated output")
    parser.add_argument("--first_frame_only", action="store_true", help="Calibrate only the first frame")
    args = parser.parse_args()

    # Set up default paths if not provided
    base_path = Path("/data/share/SoccerTrack-v2/data")
    input_video = (
        Path(args.input_video)
        if args.input_video
        else base_path / "raw" / args.match_id / f"{args.match_id}_panorama_1st_half.mp4"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else base_path
        / "interim"
        / "calibrated_videos"
        / args.match_id
        / (f"{args.match_id}_panorama.{'jpg' if args.first_frame_only else 'mp4'}")
    )
    mapx_path = base_path / "interim" / "calibrated_keypoints" / args.match_id / f"{args.match_id}_mapx.npy"
    mapy_path = base_path / "interim" / "calibrated_keypoints" / args.match_id / f"{args.match_id}_mapy.npy"
    calibrate_video(
        match_id=args.match_id,
        input_video_path=input_video,
        mapx_path=mapx_path,
        mapy_path=mapy_path,
        output_path=output_path,
        first_frame_only=args.first_frame_only,
    )


if __name__ == "__main__":
    main()

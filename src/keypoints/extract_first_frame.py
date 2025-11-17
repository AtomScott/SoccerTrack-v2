# /home/nakamura/SoccerTrack-v2/src/keypoints/extract_first_frame.py

"""
指定した video_id のパノラマ動画から「最初の1フレーム」を切り出して保存するスクリプト。

入力動画:
    /data/share/SoccerTrack-v2/data/raw/(video_id)/(video_id)_panorama.mp4

出力画像:
    /home/nakamura/SoccerTrack-v2/outputs/first_frames/(video_id)_first.jpg

使い方の例:
    uv run python -m src.keypoints.extract_first_frame --video_id 117093
"""

from __future__ import annotations

from pathlib import Path

import cv2
from loguru import logger

# 入力・出力のベースパス
RAW_BASE_DIR = Path("/data/share/SoccerTrack-v2/data/raw")
OUTPUT_BASE_DIR = Path("/home/nakamura/SoccerTrack-v2/outputs/first_frames")


def extract_first_frame(video_id: str) -> Path:
    """
    指定した video_id のパノラマ動画から最初の1フレームを切り出して保存する。

    Args:
        video_id: 例 "117093"

    Returns:
        保存した画像ファイルの Path
    """
    # 入力動画と出力画像のパスを組み立て
    video_path = RAW_BASE_DIR / video_id / f"{video_id}_panorama.mp4"
    output_path = OUTPUT_BASE_DIR / f"{video_id}_first.jpg"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")

    logger.info(f"Successfully read first frame, saving to {output_path}")

    success = cv2.imwrite(str(output_path), frame)
    if not success:
        raise RuntimeError(f"Failed to write first frame image to: {output_path}")

    logger.info(f"Saved first frame to: {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract first frame from panorama video.")
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help=(
            "Video ID (e.g., 117093). "
            "Input: /data/share/SoccerTrack-v2/data/raw/(video_id)/(video_id)_panorama.mp4 "
            "Output: /home/nakamura/SoccerTrack-v2/outputs/first_frames/(video_id)_first.jpg"
        ),
    )
    args = parser.parse_args()

    extract_first_frame(args.video_id)


if __name__ == "__main__":
    main()

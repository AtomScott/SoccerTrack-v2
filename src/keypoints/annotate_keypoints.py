# /home/nakamura/SoccerTrack-v2/src/keypoints/annotate_keypoints.py

"""
パノラマ動画の最初の1フレームに対して、ピッチ上のキーポイントを順番にクリックして
keypoints.json を作成するためのスクリプト。

- カメラは固定前提
- 入力動画: /data/share/SoccerTrack-v2/data/raw/(video_id)/(video_id)_panorama.mp4
- 出力JSON: /data/share/SoccerTrack-v2/data/raw/(video_id)/(video_id)_keypoints.json

使い方の例:
    uv run python -m src.keypoints.annotate_keypoints --video_id 117093

操作方法:
    - ウィンドウ上でマウス左クリック: 現在要求されているピッチ座標の対応点を指定
    - キーボード 's' キー: 現在のポイントを skip（JSONには記録しない）
    - キーボード 'q' キー: 途中で終了（それまでの分だけ JSON に保存される）

番号付け:
    117093_keypoints.json と同じ「ピッチ座標の並び順」で、
    リストの先頭から 1,2,3,... と番号を振っていきます。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
from loguru import logger

# ====== デフォルトベースパス ======
BASE_RAW_DIR = Path("/data/share/SoccerTrack-v2/data/raw")
# ==================================


# 117093_keypoints.json と同じ順番で並べたピッチ座標（文字列）
WORLD_KEYPOINTS_ORDER: List[str] = [
    # ゴールライン (y=68) 左端→右端
    "(0,68)",
    "(5.5,68)",
    "(11,68)",
    "(16.5,68)",
    "(22,68)",
    "(27.5,68)",
    "(33,68)",
    "(38.5,68)",
    "(44,68)",
    "(49.5,68)",
    "(52.5,68)",
    "(55.5,68)",
    "(61,68)",
    "(66.5,68)",
    "(72,68)",
    "(77.5,68)",
    "(83,68)",
    "(88.5,68)",
    "(94,68)",
    "(99.5,68)",
    "(105,68)",

    # ゴールライン (y=0) 左端→右端
    "(0,0)",
    "(5.5,0)",
    "(11,0)",
    "(16.5,0)",
    "(22,0)",
    "(27.5,0)",
    "(33,0)",
    "(38.5,0)",
    "(44,0)",
    "(49.5,0)",
    "(52.5,0)",
    "(55.5,0)",
    "(61,0)",
    "(66.5,0)",
    "(72,0)",
    "(77.5,0)",
    "(83,0)",
    "(88.5,0)",
    "(94,0)",
    "(99.5,0)",
    "(105,0)",

    # 左タッチライン上のいくつかのポイント
    "(0,54.16)",
    "(0,43.16)",
    "(0,37.66)",
    "(0,30.34)",
    "(0,24.84)",
    "(0,13.84)",

    # 左ペナルティエリア / PA 付近
    "(16.5,13.84)",
    "(16.5,54.16)",
    "(5.5,43.16)",
    "(5.5,24.84)",

    # 右タッチライン / PA 付近
    "(105,54.16)",
    "(88.5,54.16)",
    "(88.5,13.84)",
    "(99.5,24.84)",
    "(99.5,43.16)",
    "(105,43.16)",
    "(105,37.66)",
    "(105,30.34)",
    "(105,24.84)",
    "(105,13.84)",

    # センター付近
    "(52.5,43.15)",
    "(52.5,34)",
    "(52.5,24.85)",
]


@dataclass
class ClickState:
    last_click: Optional[Tuple[int, int]] = None


def mouse_callback(event, x, y, flags, param):
    """
    マウスコールバック: 左クリックされた座標を ClickState に記録する。
    """
    state: ClickState = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state.last_click = (x, y)
        logger.info(f"Clicked at (x={x}, y={y})")


def draw_existing_points(frame, points: List[Tuple[str, Tuple[int, int]]]) -> None:
    """
    すでに確定しているキーポイントをフレーム上に描画する。
    points: [(world_str, (x,y)), ...]
    """
    h, w = frame.shape[:2]
    base_scale = max(w / 4000.0, 0.5)
    font_scale = 0.6 * base_scale
    thickness = max(int(2 * base_scale), 1)

    for idx, (world_str, (x, y)) in enumerate(points, start=1):
        cx, cy = int(x), int(y)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), thickness=-1)  # 赤い点
        text = str(idx)
        cv2.putText(
            frame,
            text,
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )


def annotate_keypoints_for_video(
    video_id: str,
    world_keypoints_order: List[str] = WORLD_KEYPOINTS_ORDER,
) -> Path:
    """
    与えられた video_id のパノラマ動画の最初のフレームに対して、
    WORLD_KEYPOINTS_ORDER の順番でポイントをクリックし、
    JSON に { "(x_pitch,y_pitch)": [x_img, y_img], ... } 形式で保存する。

    Args:
        video_id: 例 "117093"

    Returns:
        出力JSONのPath
    """
    raw_dir = BASE_RAW_DIR / video_id
    video_path = raw_dir / f"{video_id}_panorama.mp4"
    output_json = raw_dir / f"{video_id}_keypoints_1.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")

    logger.info("Successfully read first frame for annotation.")

    # ウィンドウとマウスコールバックのセットアップ
    window_name = f"Keypoint Annotation - {video_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    click_state = ClickState()
    cv2.setMouseCallback(window_name, mouse_callback, click_state)

    annotated_points: Dict[str, List[float]] = {}
    confirmed_points_for_drawing: List[Tuple[str, Tuple[int, int]]] = []

    h, w = frame.shape[:2]
    base_scale = max(w / 4000.0, 0.5)
    info_font_scale = 0.7 * base_scale
    info_thickness = max(int(2 * base_scale), 1)

    logger.info("操作説明: 左クリックでポイント選択, 's' キーでスキップ, 'q' キーで終了")

    for idx, world_str in enumerate(world_keypoints_order, start=1):
        click_state.last_click = None  # リセット
        while True:
            # 描画用のコピーを作成
            vis = frame.copy()

            # すでに確定している点を描画
            draw_existing_points(vis, confirmed_points_for_drawing)

            # 画面下部に現在のインデックス & ピッチ座標を表示
            info_text = f"{idx}/{len(world_keypoints_order)} : Select point for {world_str}  (click: select, 's': skip, 'q': quit)"
            y_pos = h - 20
            cv2.putText(
                vis,
                info_text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                info_font_scale,
                (255, 255, 255),
                info_thickness,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(50) & 0xFF

            if key == ord("q"):
                logger.warning("User requested quit. Stopping annotation.")
                cv2.destroyWindow(window_name)
                # ここまでの annotated_points だけ保存して return
                output_json.parent.mkdir(parents=True, exist_ok=True)
                with output_json.open("w") as f:
                    json.dump(annotated_points, f, indent=4)
                logger.info(f"Partial keypoints saved to: {output_json}")
                return output_json

            if key == ord("s"):
                # 現在のポイントをスキップ
                logger.info(f"Skipping point {idx}: {world_str}")
                break  # 次の world_str へ

            if click_state.last_click is not None:
                # クリックされた座標を採用
                cx, cy = click_state.last_click
                logger.info(f"Point {idx} ({world_str}) annotated at image coords ({cx}, {cy})")
                annotated_points[world_str] = [float(cx), float(cy)]
                confirmed_points_for_drawing.append((world_str, (cx, cy)))
                break  # 次の world_str へ

        # 次のポイントへ

    cv2.destroyWindow(window_name)

    # 全ポイント or スキップを反映した JSON を保存
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(annotated_points, f, indent=4)

    logger.info(f"All annotations finished. Saved keypoints to: {output_json}")
    return output_json


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Annotate pitch keypoints on first frame of panorama video.")
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="Video ID (e.g., 117093). "
        "Input video is assumed at /data/share/SoccerTrack-v2/data/raw/(video_id)/(video_id)_panorama.mp4",
    )
    args = parser.parse_args()

    annotate_keypoints_for_video(video_id=args.video_id)


if __name__ == "__main__":
    main()

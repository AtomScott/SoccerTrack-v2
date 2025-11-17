# /home/nakamura/SoccerTrack-v2/src/keypoints/visualize_keypoints.py
# uv run python -m src.keypoints.visualize_keypoints

import json
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
from loguru import logger

# ====== デフォルトパス（ここを編集して他の試合にも使えるようにする） ======
DEFAULT_KEYPOINTS_PATH = Path("/data/share/SoccerTrack-v2/data/raw/118577/118577_keypoints.json")
DEFAULT_VIDEO_PATH = Path("/data/share/SoccerTrack-v2/data/raw/118578/118578_panorama.mp4")
DEFAULT_OUTPUT_DIR = Path("/home/nakamura/SoccerTrack-v2/outputs/keypoints")
DEFAULT_OUTPUT_FILENAME = "118578_keypoints_overlay_1.jpg"
# ===============================================================

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


def load_keypoints_json(keypoints_path: Path) -> Dict[str, List[float]]:
    """
    キーポイント JSON を読み込む。
    JSON の順番はそのまま保持される（Python 3.7+ では dict が順序付き）。

    JSON 形式:
    {
        "(0,68)": [111.399, 598.321],
        "(5.5,68)": [161.899, 605.488],
        ...
    }

    Returns:
        dict: { "world_coord_str": [x_img, y_img], ... }
    """
    logger.info(f"Loading keypoints from {keypoints_path}")
    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints JSON not found: {keypoints_path}")

    with keypoints_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Keypoints JSON must be a dict, got {type(data)}")

    return data


def parse_world_coord(coord_str: str) -> Tuple[float, float]:
    """
    "(0,68)" のような文字列を (0.0, 68.0) に変換する。
    world 座標は今回は描画には使わないが、index の順番確認や今後のためにパースしておく。
    """
    # 括弧を取り除いてからカンマ分割
    stripped = coord_str.strip().strip("()")
    x_str, y_str = stripped.split(",")
    return float(x_str), float(y_str)


def draw_keypoints_on_frame(
    frame,
    keypoints: Dict[str, List[float]],
    circle_radius: int = 6,
    circle_color: Tuple[int, int, int] = (0, 0, 255),  # BGR: 赤
    text_color: Tuple[int, int, int] = (0, 255, 0),    # BGR: 緑
    world_keypoints_order: List[str] | None = None,    # ★追加
) -> "cv2.Mat":
    """
    フレームに keypoints を 1,2,3,... の番号付きで描画する。

    Args:
        frame: OpenCV 画像 (BGR)
        keypoints: { "(world_x,world_y)": [x_img, y_img], ... }
        circle_radius: 円の半径
        circle_color: 円の色 (B,G,R)
        text_color: 文字の色 (B,G,R)
        world_keypoints_order: 番号付けに使う「ピッチ座標の順番」。
            None の場合は keypoints の順番をそのまま使う。
    """
    annotated = frame.copy()

    h, w = annotated.shape[:2]
    base_scale = max(w / 4000.0, 0.5)
    font_scale = 0.7 * base_scale
    thickness = max(int(2 * base_scale), 1)

    # ★順番を決める
    if world_keypoints_order is None:
        # 旧仕様：JSON の順番そのまま
        ordered_world_keys = list(keypoints.keys())
    else:
        # 固定の WORLD_KEYPOINTS_ORDER に従う
        ordered_world_keys = world_keypoints_order

    # ★この順番で 1,2,3,... を振る
    for idx, world_str in enumerate(ordered_world_keys, start=1):
        if world_str not in keypoints:
            # その世界座標が JSON に無い場合は「欠番」扱いで何も描画しない
            continue

        img_xy = keypoints[world_str]
        if not isinstance(img_xy, (list, tuple)) or len(img_xy) != 2:
            logger.warning(f"Invalid image coords for key {world_str}: {img_xy}")
            continue

        x_img, y_img = img_xy
        cx, cy = int(round(x_img)), int(round(y_img))

        # 円
        cv2.circle(annotated, (cx, cy), circle_radius, circle_color, thickness=-1)

        # 番号
        text = str(idx)
        text_org = (cx + 5, cy - 5)
        cv2.putText(
            annotated,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        # デバッグログ（任意）
        try:
            wx, wy = parse_world_coord(world_str)
            logger.debug(f"Idx {idx}: world=({wx}, {wy}), image=({x_img}, {y_img})")
        except Exception:
            logger.debug(f"Idx {idx}: world_str={world_str}, image=({x_img}, {y_img})")

    return annotated



def visualize_keypoints_on_first_frame(
    keypoints_path: Path,
    video_path: Path,
    output_dir: Path,
    output_filename: str,
) -> Path:
    """
    入力動画の最初のフレームに keypoints を描画し、画像として保存する。

    Args:
        keypoints_path: キーポイント JSON のパス
        video_path: 入力動画のパス
        output_dir: 出力ディレクトリ
        output_filename: 出力ファイル名（例: '117093_keypoints_overlay.jpg'）

    Returns:
        保存した画像のパス
    """
    keypoints_path = Path(keypoints_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    # 1. キーポイント読み込み
    keypoints = load_keypoints_json(keypoints_path)

    # 2. 動画の最初のフレームを取得
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")

    logger.info("Successfully read first frame, drawing keypoints...")

    # 3. キーポイントを描画
    annotated = draw_keypoints_on_frame(
        frame,
        keypoints,
        world_keypoints_order=WORLD_KEYPOINTS_ORDER,
    )

    # 4. 保存
    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        raise RuntimeError(f"Failed to write annotated image to: {output_path}")

    logger.info(f"Saved annotated keypoints image to: {output_path}")
    return output_path


def main():
    """
    コマンドラインから実行できるようにするエントリポイント。
    引数を変えたいときはここを編集 or CLI から指定する。
    """
    import argparse

    parser = argparse.ArgumentParser(description="Visualize keypoints on the first frame of a video.")
    parser.add_argument(
        "--keypoints_path",
        type=str,
        default=str(DEFAULT_KEYPOINTS_PATH),
        help=f"Path to keypoints JSON file (default: {DEFAULT_KEYPOINTS_PATH})",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help=f"Path to input video (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save annotated image (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output filename (default: {DEFAULT_OUTPUT_FILENAME})",
    )

    args = parser.parse_args()

    visualize_keypoints_on_first_frame(
        keypoints_path=Path(args.keypoints_path),
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSRアノテーションCSVを用いて動画に描画し、ffmpegで軽量エンコードして保存するスクリプト。
- 進捗バー: tqdm
- left: 青, right: 白, GK: 黄 + "goalkeeper" ラベル
- jersey が空なら id を表示
- 入出力パスはコード内の定数で固定
"""

from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import cv2
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ====== 入出力パス（必要に応じて書き換えてください）======
VIDEO_PATH  = Path("/data/share/SoccerTrack-v2/data/interim/117093/117093_calibrated_panorama_1st_half.mp4")
CSV_PATH    = Path("/data/share/SoccerTrack-v2/data/interim/117093/117093_player_gsr_1st_half_calibrated.csv")
OUTPUT_PATH = Path("/data/share/SoccerTrack-v2/data/interim/117093/117093_player_gsr_1st_half_calibrated_annotated.mp4")
# ===========================================================

# エンコード設定（小さくしたいほど CRF を大きく：23(標準) → 26〜30(軽量)）
FFMPEG_CRF = "26"
FFMPEG_PRESET = "medium"  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
FFMPEG_PIXEL_FMT = "yuv420p"  # 互換性の高いピクセルフォーマット

THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# OpenCVはBGR
COLOR_BLUE   = (255, 0, 0)     # left
COLOR_WHITE  = (255, 255, 255) # right
COLOR_YELLOW = (0, 255, 255)   # GK
COLOR_GRAY   = (200, 200, 200) # unknown
COLOR_BLACK  = (0, 0, 0)

def put_text_with_bg(
    img,
    text: str,
    org,
    text_color,
    bg_color=COLOR_BLACK,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=FONT_SCALE,
    thickness=FONT_THICKNESS,
    padding=3,
):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x1 = max(0, x - padding)
    y1 = max(0, y - th - baseline - padding)
    x2 = min(img.shape[1] - 1, x + tw + padding)
    y2 = min(img.shape[0] - 1, y + padding)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x, y - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)

def annotate_and_write(temp_out: Path):
    # CSV 読み込み
    cols = [
        "frame","id","supercategory","category_id","role","jersey","team",
        "bb_left","bb_top","bb_width","bb_height","conf","x","y","z",
    ]
    df = pd.read_csv(CSV_PATH, usecols=cols)

    # 型整備
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int)
    df["id"]    = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(int)
    for c in ["bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # フレーム→アノテーション行のマップ
    by_frame = defaultdict(list)
    for row in df.itertuples(index=False):
        by_frame[int(row.frame)].append(row)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"ERROR: failed to open video: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # まずは素のmp4vで書き出し
    temp_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(temp_out), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_for_bar = total_frames if total_frames and total_frames > 0 else None

    frame_idx = 0
    with tqdm(total=total_for_bar, unit="f", desc=f"Annotating {VIDEO_PATH.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rows = by_frame.get(frame_idx, [])
            if rows:
                for r in rows:
                    x1 = int(round(r.bb_left))
                    y1 = int(round(r.bb_top))
                    w  = max(1, int(round(r.bb_width  if pd.notna(r.bb_width)  else 1)))
                    h  = max(1, int(round(r.bb_height if pd.notna(r.bb_height) else 1)))
                    x2 = x1 + w
                    y2 = y1 + h

                    # クリップ
                    x1 = max(0, min(x1, width  - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width  - 1))
                    y2 = max(0, min(y2, height - 1))

                    role = (str(r.role) if pd.notna(r.role) else "").strip().lower()
                    team = (str(r.team) if pd.notna(r.team) else "").strip().lower()

                    if role == "goalkeeper":
                        color = COLOR_YELLOW
                        label = "goalkeeper"
                    else:
                        if team == "left":
                            color = COLOR_BLUE
                        elif team == "right":
                            color = COLOR_WHITE
                        else:
                            color = COLOR_GRAY
                        jersey = (str(r.jersey).strip() if pd.notna(r.jersey) else "")
                        label = jersey if jersey else str(r.id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
                    tx, ty = x1, max(0, y1 - 5)
                    put_text_with_bg(frame, label, (tx, ty), text_color=color, bg_color=COLOR_BLACK)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def reencode_with_ffmpeg(src: Path, dst: Path) -> bool:
    """ffmpeg で H.264 再エンコード（軽量化）。成功時 True。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", FFMPEG_CRF,
        "-pix_fmt", FFMPEG_PIXEL_FMT,
        "-movflags", "+faststart",  # ストリーミング開始を早く
        str(dst),
    ]
    try:
        print("Re-encoding with ffmpeg for smaller size...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"WARNING: ffmpeg re-encoding failed: {e}", file=sys.stderr)
        return False

def main():
    if not VIDEO_PATH.exists():
        print(f"ERROR: video not found: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CSV_PATH.exists():
        print(f"ERROR: csv not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # 一時ファイルに一旦書き出し（その後 ffmpeg 再エンコード）
    tmp_dir = Path(tempfile.mkdtemp(prefix="annotate_"))
    raw_out = tmp_dir / (OUTPUT_PATH.stem + "_raw.mp4")

    try:
        annotate_and_write(raw_out)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if ffmpeg_available():
            ok = reencode_with_ffmpeg(raw_out, OUTPUT_PATH)
            if ok:
                print(f"Saved annotated & compressed video to: {OUTPUT_PATH}")
            else:
                # ffmpeg失敗時は非圧縮のまま退避
                fallback = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_raw_fallback.mp4")
                shutil.move(str(raw_out), str(fallback))
                print(f"Saved raw annotated video (no compression) to: {fallback}")
        else:
            # ffmpeg が無ければそのままコピー
            fallback = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_raw.mp4")
            shutil.move(str(raw_out), str(fallback))
            print(f"ffmpeg not found. Saved raw annotated video to: {fallback}")

    finally:
        # 一時ディレクトリの残骸を片付け（raw_out を移動済みなら残りを削除）
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()

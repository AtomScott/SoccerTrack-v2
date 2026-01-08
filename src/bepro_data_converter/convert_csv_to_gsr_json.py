#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
CSV(検出/ピッチ座標/校正キーポイント) -> GSR用JSON 変換スクリプト（distorted版）

- 先頭 N 秒のみ出力 (--first-seconds を指定した場合)
- 指定しなければ CSV に含まれる全フレームを変換
- gsr と pitch CSV のフレームずれを --pitch-frame-offset で補正
- ball を除外し、数値ID(選手)のみを対象に結合
- bbox_image を生成
- pitch_plane の正規化座標(u,v)→ピッチ中心原点[m]へ
- distorted キーポイントJSON + nbjw_calibの対応表から pitch.lines を各フレームに付与（不変）

想定パス（デフォルト）:
  base_interim = /data/share/SoccerTrack-v2/data/interim/{video_id}
  base_raw     = /data/share/SoccerTrack-v2/data/raw/{video_id}

  detections:
    {video_id}_player_gsr_{half}_half_distorted.csv
      例: 132831_player_gsr_1st_half_distorted.csv

  pitch:
    {video_id}_pitch_plane_coordinates_{half}_half.csv
      例: 132831_pitch_plane_coordinates_1st_half.csv

  keypoints-json (distorted):
    /data/share/SoccerTrack-v2/data/raw/{video_id}/{video_id}_keypoints.json

  out:
    /data/share/SoccerTrack-v2/data/interim/{video_id}/Labels-GameState_{video_id}_distorted_Full.json

使い方:
  python3 -m data.interim.convert_csv_to_gsr_json 132877 1st
  python3 -m data.interim.convert_csv_to_gsr_json 117092 2nd --first-seconds 30
  ※ half は detections/pitch のファイル推定に使われますが、
     デフォルト出力パスはどちらも「*_distorted_Full.json」になります。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import pandas as pd
import numpy as np

# ==========================
# 定数: カテゴリとライン名
# ==========================

CATEGORY_BY_ROLE = {
    "player": 1,
    "goalkeeper": 2,
    "referee": 3,
    "ball": 4,
}

CATEGORIES_SECTION = [
    {"supercategory": "object", "id": 1, "name": "player"},
    {"supercategory": "object", "id": 2, "name": "goalkeeper"},
    {"supercategory": "object", "id": 3, "name": "referee"},
    {"supercategory": "object", "id": 4, "name": "ball"},
    {
        "supercategory": "pitch",
        "id": 5,
        "name": "pitch",
        "lines": [
            "Big rect. left bottom",
            "Big rect. left main",
            "Big rect. left top",
            "Big rect. right bottom",
            "Big rect. right main",
            "Big rect. right top",
            "Circle central",
            "Circle left",
            "Circle right",
            "Goal left crossbar",
            "Goal left post left",
            "Goal left post right",
            "Goal right crossbar",
            "Goal right post left",
            "Goal right post right",
            "Middle line",
            "Side line bottom",
            "Side line left",
            "Side line right",
            "Side line top",
            "Small rect. left bottom",
            "Small rect. left main",
            "Small rect. left top",
            "Small rect. right bottom",
            "Small rect. right main",
            "Small rect. right top",
        ],
    },
    {"supercategory": "camera", "id": 6, "name": "camera"},
    {"supercategory": "object", "id": 7, "name": "other"},
]


# ==================================================
# nbjw_calib: キーポイント→ラインの定義（1-based index）
# ==================================================
def kp_to_line() -> Dict[str, List[int]]:
    return {
        "Big rect. left bottom": [24, 68, 25],
        "Big rect. left main": [5, 64, 31, 46, 34, 66, 25],
        "Big rect. left top": [4, 62, 5],
        "Big rect. right bottom": [26, 69, 27],
        "Big rect. right main": [6, 65, 33, 56, 36, 67, 26],
        "Big rect. right top": [6, 63, 7],
        "Circle central": [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
        "Circle left": [31, 37, 47, 41, 34],
        "Circle right": [33, 40, 55, 44, 36],
        "Goal left crossbar": [16, 12],
        "Goal left post left": [16, 17],
        "Goal left post right": [12, 13],
        "Goal right crossbar": [15, 19],
        "Goal right post left": [15, 14],
        "Goal right post right": [19, 18],
        "Middle line": [2, 32, 51, 35, 29],
        "Side line bottom": [28, 70, 71, 29, 72, 73, 30],
        "Side line left": [1, 4, 8, 13, 17, 20, 24, 28],
        "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
        "Side line top": [1, 58, 59, 2, 60, 61, 3],
        "Small rect. left bottom": [20, 21],
        "Small rect. left main": [9, 21],
        "Small rect. left top": [8, 9],
        "Small rect. right bottom": [22, 23],
        "Small rect. right main": [10, 22],
        "Small rect. right top": [10, 11],
    }


# ==================================================
# utils_calib: ワールド座標の定義（未センタリング値を保持）
# ==================================================
BASE_WORLD = [
    [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
    [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
    [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
    [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
    [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
    [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
    [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
    [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
    [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
    [88.5, 34.], [94., 34.]
]  # 57

AUX_WORLD = [
    [5.5, 0], [16.5, 0], [88.5, 0], [99.5, 0], [5.5, 13.84], [99.5, 13.84], [16.5, 24.84],
    [88.5, 24.84], [16.5, 43.16], [88.5, 43.16], [5.5, 54.16], [99.5, 54.16], [5.5, 68],
    [16.5, 68], [88.5, 68], [99.5, 68]
]  # 16

KP_ALL_ABS: List[Tuple[float, float]] = [tuple(xy) for xy in (BASE_WORLD + AUX_WORLD)]  # 1..73対応


# ==========================
# 便利関数
# ==========================

def half_step_center(value: float) -> float:
    """center値を0.5刻みに丸める（例JSONの体裁に寄せる）。"""
    return float(np.round(value * 2) / 2.0)


def build_track_id_map(unique_ids: List[int]) -> Dict[int, int]:
    """元ID(int) -> 1..N の安定マップを作る"""
    sorted_ids = sorted(unique_ids)
    return {orig_id: i + 1 for i, orig_id in enumerate(sorted_ids)}


def category_id_from_role(role: str) -> int:
    role = (role or "").strip().lower()
    return CATEGORY_BY_ROLE.get(role, 7)  # unknown -> other(7)


def to_bbox_image(row: pd.Series) -> Dict[str, Any]:
    x = float(row["bb_left"])
    y = float(row["bb_top"])
    w = float(row["bb_width"])
    h = float(row["bb_height"])
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    return {
        "x": int(round(x)),
        "y": int(round(y)),
        "x_center": half_step_center(x_center),
        "y_center": half_step_center(y_center),
        "w": int(round(w)),
        "h": int(round(h)),
    }


def normalized_to_pitch_xy(u: float, v: float,
                           pitch_length: float, pitch_width: float) -> Tuple[float, float]:
    """正規化(左上原点) -> ピッチ中心原点[m]"""
    x_m = (u - 0.5) * pitch_length
    y_m = (v - 0.5) * pitch_width
    return float(x_m), float(y_m)


def build_bbox_pitch_from_normalized(u: float, v: float,
                                     pitch_length: float, pitch_width: float) -> Dict[str, Any]:
    """底辺中心のみ → left/right は同値を複写"""
    xm, ym = normalized_to_pitch_xy(u, v, pitch_length, pitch_width)
    return {
        "x_bottom_left": xm, "y_bottom_left": ym,
        "x_bottom_right": xm, "y_bottom_right": ym,
        "x_bottom_middle": xm, "y_bottom_middle": ym,
    }


# ==========================
# ピッチ lines 生成
# ==========================

def parse_keypoints_json(path: Path) -> Dict[Tuple[float, float], Tuple[float, float]]:
    """
    distorted キーポイントJSONを読み込み：
      key: "(x,y)" (未センタリング[m])
      val: [px, py] 画像座標 (pixel)
    を { (x,y): (px,py) } にする
    """
    d = json.loads(Path(path).read_text())
    out: Dict[Tuple[float, float], Tuple[float, float]] = {}
    for k, v in d.items():
        k = k.strip().lstrip("(").rstrip(")")
        xs, ys = k.split(",")
        xw = float(xs.strip())
        yw = float(ys.strip())
        px = float(v[0])
        py = float(v[1])
        out[(xw, yw)] = (px, py)
    return out


def find_pixel_for_world(world_xy: Tuple[float, float],
                         kp_map: Dict[Tuple[float, float], Tuple[float, float]],
                         tol: float = 0.02) -> Optional[Tuple[float, float]]:
    """
    world座標 (m) に対する画素座標を校正マップから取得。
    まずは完全一致、無ければ ±tol[m] の最近傍を探索。
    """
    if world_xy in kp_map:
        return kp_map[world_xy]
    xw, yw = world_xy
    best = None
    best_dist = 1e9
    for (xa, ya), (px, py) in kp_map.items():
        dist = (xa - xw) ** 2 + (ya - yw) ** 2
        if dist < best_dist:
            best_dist = dist
            best = (px, py)
    if best is not None and np.sqrt(best_dist) <= tol:
        return best
    return None


def build_pitch_lines_for_image(im_w: int, im_h: int,
                                kp_map: Dict[Tuple[float, float], Tuple[float, float]]
                                ) -> Dict[str, List[Dict[str, float]]]:
    """
    1画像ぶんの lines を生成（全フレーム不変なので使い回し可）
    - kp_to_line の index は 1-based
    - KP_ALL_ABS の world座標(未センタリング)で校正JSONを引く
    - 画像サイズで正規化 (0..1)
    """
    mapping = kp_to_line()
    lines_out: Dict[str, List[Dict[str, float]]] = {}
    for line_name, idx_list in mapping.items():
        pts_norm: List[Dict[str, float]] = []
        for idx in idx_list:
            world_xy = KP_ALL_ABS[idx - 1]  # 1-based -> 0-based
            pix = find_pixel_for_world(world_xy, kp_map)
            if pix is None:
                continue
            px, py = pix
            pts_norm.append({
                "x": float(px) / float(im_w),
                "y": float(py) / float(im_h),
            })
        lines_out[line_name] = pts_norm
    return lines_out


# ==========================
# メイン
# ==========================

def main():
    ap = argparse.ArgumentParser(
        description="SoccerTrack-v2 の GSR CSV を STC-GS 形式JSONに変換するスクリプト（distorted版）"
    )

    # 必須: video_id / half
    ap.add_argument("video_id", type=str, help="例: 117092")
    ap.add_argument("half", type=str, choices=["1st", "2nd"], help="1st or 2nd")

    # パスの上書き（指定しなければ標準レイアウトを使用）
    ap.add_argument(
        "--detections",
        type=Path,
        help="player_gsr_*_distorted.csv のパス（未指定なら自動推定）",
    )
    ap.add_argument(
        "--pitch",
        type=Path,
        help="pitch_plane_coordinates_*_half.csv のパス（未指定なら自動推定）",
    )
    ap.add_argument(
        "--keypoints-json",
        type=Path,
        help="distorted *_keypoints.json のパス（未指定なら自動推定）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="出力JSONパス（未指定なら Labels-GameState_{video_id}_distorted_Full.json）",
    )

    # 画像関連
    ap.add_argument("--im-width", type=int, default=3840)
    ap.add_argument("--im-height", type=int, default=1504)
    # ap.add_argument("--im-width", type=int, default=4096)
    # ap.add_argument("--im-height", type=int, default=1080)
    ap.add_argument("--frame-rate", type=int, default=25)

    # ピッチ寸法[m]
    ap.add_argument("--pitch-length", type=float, default=105.0)
    ap.add_argument("--pitch-width", type=float, default=68.0)

    # info メタ
    ap.add_argument("--seq-name", default=None,
                    help="デフォルト: CLPD-{video_id}")
    ap.add_argument("--game-id", default="7")
    ap.add_argument("--clip-id", default="1")
    ap.add_argument("--num-tracklets", default=None)
    ap.add_argument("--action-position", default="0")
    ap.add_argument("--action-class", default="Unknown")
    ap.add_argument("--visibility", default="visible")
    ap.add_argument("--game-time-start", default="1 - 00:00")
    ap.add_argument("--game-time-stop", default="1 - 00:30")
    ap.add_argument("--clip-start", default="0")
    ap.add_argument("--clip-stop", default="30000")
    ap.add_argument("--im-dir", default="img1")
    ap.add_argument("--im-ext", default=".jpg")
    ap.add_argument(
        "--id-prefix",
        default="3",
        help="image_id/annotation id の接頭数値文字列（任意）",
    )

    # オフセット & 先頭N秒
    ap.add_argument(
        "--pitch-frame-offset",
        type=int,
        default=-251,
        help="ピッチCSVのframeに加えるオフセット（gsr(0)とpitch(251)→-251が既定）",
    )
    ap.add_argument(
        "--first-seconds",
        type=float,
        default=None,
        help="先頭から出力する秒数。指定しなければ全フレームを変換。",
    )

    args = ap.parse_args()

    video_id = args.video_id
    half = args.half
    half_tag = f"{half}_half"

    # ---- パス決定 ----
    base_interim = Path(f"/data/share/SoccerTrack-v2/data/interim/{video_id}")
    base_raw = Path(f"/data/share/SoccerTrack-v2/data/raw/{video_id}")

    det_path = args.detections or (
        base_interim / f"{video_id}_player_gsr_{half_tag}_distorted.csv"
    )
    pitch_path = args.pitch or (
        base_interim / f"{video_id}_pitch_plane_coordinates_{half_tag}.csv"
    )
    # distorted キーポイント
    kp_path = args.keypoints_json or (
        base_raw / f"{video_id}_keypoints.json"
    )
    # 出力は Full 固定 (上書きが嫌なら --out で変更)
    out_path = args.out or (
        base_interim / f"Labels-GameState_{video_id}_distorted_{half}.json"
    )

    # ---- CSV 読み込み ----
    det_df = pd.read_csv(det_path)
    pitch_df = pd.read_csv(pitch_path)

    # frame
    det_df["frame"] = det_df["frame"].astype(int)
    pitch_df["frame"] = pitch_df["frame"].astype(int)

    # 選手のみ（ball等は除外, 数値IDのみ残す）
    det_df["id"] = pd.to_numeric(det_df["id"], errors="coerce")
    pitch_df["id"] = pd.to_numeric(pitch_df["id"], errors="coerce")
    det_df = det_df.dropna(subset=["id"]).copy()
    pitch_df = pitch_df.dropna(subset=["id"]).copy()
    det_df["id"] = det_df["id"].astype(int)
    pitch_df["id"] = pitch_df["id"].astype(int)

    # ピッチフレーム補正
    if int(args.pitch_frame_offset) != 0:
        pitch_df["frame"] = pitch_df["frame"] + int(args.pitch_frame_offset)

    # 先頭 N 秒だけに絞るかどうか
    if args.first_seconds is not None and args.first_seconds > 0:
        n_frames_wanted = int(round(args.frame_rate * float(args.first_seconds)))
        all_frames_sorted = sorted(det_df["frame"].unique().tolist())
        keep_frames = set(all_frames_sorted[:n_frames_wanted])
        det_df = det_df[det_df["frame"].isin(keep_frames)].copy()
        pitch_df = pitch_df[pitch_df["frame"].isin(keep_frames)].copy()
    else:
        # 全フレームを使う場合、pitch側はdetに存在するframeだけに絞る
        keep_frames = set(det_df["frame"].unique().tolist())
        pitch_df = pitch_df[pitch_df["frame"].isin(keep_frames)].copy()

    # ピッチ座標 (u,v)
    pitch_pts = pitch_df.rename(columns={"x": "u", "y": "v"})[["frame", "id", "u", "v"]]

    # track_id 割当
    unique_ids = det_df["id"].unique().tolist()
    track_map = build_track_id_map(unique_ids)

    # 出力対象フレーム
    frames = sorted(det_df["frame"].unique().tolist())
    seq_length = len(frames)

    # images セクション
    images: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames, start=1):
        images.append({
            "is_labeled": True,
            "image_id": f"{args.id_prefix}{i:06d}",
            "file_name": f"{i:06d}{args.im_ext}",
            "height": args.im_height,
            "width": args.im_width,
            "has_labeled_person": True,
            "has_labeled_pitch": True,
            "has_labeled_camera": True,
            "ignore_regions_y": [],
            "ignore_regions_x": [],
        })

    frame_to_image_id = {
        fr: f"{args.id_prefix}{i:06d}" for i, fr in enumerate(frames, start=1)
    }

    # distorted キーポイントから pitch.lines を構築（全フレーム共通）
    kp_map = parse_keypoints_json(kp_path)
    lines_template = build_pitch_lines_for_image(
        args.im_width, args.im_height, kp_map
    )

    # アノテーション配列
    annotations: List[Dict[str, Any]] = []

    # まず各 image に pitch アノテーション（lines付）
    for i, fr in enumerate(frames, start=1):
        image_id = frame_to_image_id[fr]
        pitch_ann_id = f"{args.id_prefix}{i:06d}P"
        annotations.append({
            "id": pitch_ann_id,
            "image_id": image_id,
            "video_id": str(args.clip_id),
            "supercategory": "pitch",
            "category_id": 5,
            "lines": lines_template,
        })

    # 検出とピッチ(u,v)を結合し、プレイヤー注釈を作成
    det_merged = det_df.merge(pitch_pts, on=["frame", "id"], how="left")

    for fr in frames:
        sub = det_merged[det_merged["frame"] == fr].sort_values(by=["id"]).reset_index(drop=True)
        image_id = frame_to_image_id[fr]
        for row_idx, row in sub.iterrows():
            ann_local_idx = row_idx + 1
            ann_id = f"{args.id_prefix}{fr:06d}{ann_local_idx:02d}"
            track_id = track_map[int(row["id"])]
            role = str(row.get("role", "") or "")
            team = str(row.get("team", "") or "")

            jv = row.get("jersey")
            if pd.isna(jv):
                jersey = ""
            else:
                try:
                    jersey = str(int(jv))
                except Exception:
                    jersey = str(jv)

            bbox_image = to_bbox_image(row)
            if pd.notna(row.get("u")) and pd.notna(row.get("v")):
                bbox_pitch = build_bbox_pitch_from_normalized(
                    float(row["u"]),
                    float(row["v"]),
                    args.pitch_length,
                    args.pitch_width,
                )
                bbox_pitch_raw = dict(bbox_pitch)
            else:
                bbox_pitch = {
                    "x_bottom_left": None,
                    "y_bottom_left": None,
                    "x_bottom_right": None,
                    "y_bottom_right": None,
                    "x_bottom_middle": None,
                    "y_bottom_middle": None,
                }
                bbox_pitch_raw = dict(bbox_pitch)

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "track_id": track_id,
                "supercategory": "object",
                "category_id": category_id_from_role(role),
                "attributes": {
                    "role": role if role else "player",
                    "jersey": jersey,
                    "team": team,
                },
                "bbox_image": bbox_image,
                "bbox_pitch": bbox_pitch,
                "bbox_pitch_raw": bbox_pitch_raw,
            })

    # info
    num_tracklets = args.num_tracklets or str(len(unique_ids))
    seq_name = args.seq_name or f"CLPD-{video_id}"

    info = {
        "version": "1.3",
        "game_id": str(args.game_id),
        "id": str(args.clip_id),
        "num_tracklets": str(num_tracklets),
        "action_position": str(args.action_position),
        "action_class": str(args.action_class),
        "visibility": str(args.visibility),
        "game_time_start": str(args.game_time_start),
        "game_time_stop": str(args.game_time_stop),
        "clip_start": str(args.clip_start),
        "clip_stop": str(args.clip_stop),
        "name": str(seq_name),
        "im_dir": str(args.im_dir),
        "frame_rate": int(args.frame_rate),
        "seq_length": int(seq_length),
        "im_ext": str(args.im_ext),
    }

    out_obj = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES_SECTION,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=4)

    sel_sec = args.first_seconds if args.first_seconds is not None else "all"
    print(
        f"Saved: {out_path} "
        f"(selected_seconds={sel_sec}, "
        f"exported_frames={len(frames)}, "
        f"annotations={len(annotations)}, "
        f"unique_ids={len(unique_ids)})"
    )


if __name__ == "__main__":
    main()

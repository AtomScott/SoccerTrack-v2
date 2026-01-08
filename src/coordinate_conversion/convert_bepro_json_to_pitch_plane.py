#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bepro形式のトラッキングJSON（前後半2ファイル）とメタデータJSONから、
XML版と同じ形式の pitch_plane_coordinates_*.csv を出力するスクリプト。

出力CSV形式（既存 convert_raw_to_pitch_plane と同じ）:
    frame, match_time, event_period, ball_status, id, x, y, teamId

- トラッキングJSON:
    {
      "1": [ { "object": "PLAYER", "player_id": 520168, "match_time": 40, "x": 96.49, "y": 33.92, "speed": 0.05 }, ... ],
      "2": [ ... ],
      ...
    }
- メタデータJSON:
    {
      "home_team": { "team_id": 15370, "players": [ { "player_id": 520184, ... }, ... ] },
      "away_team": { "team_id": 30798, "players": [ { "player_id": 520168, ... }, ... ] },
      "ground_width": 105,
      "ground_height": 68,
      ...
    }

座標系:
- JSON内の x, y は [0, ground_width], [0, ground_height] のメートル値。
- 出力CSVの x, y は XML版と合わせて [0,1] に正規化する:
    x_norm = x / ground_width
    y_norm = y / ground_height

使い方の例:
    python convert_bepro_json_to_pitch_plane.py \
        --match-id 132831 \
        --first-json  "/data/share/SoccerTrack-v2/raw_bepro_from_gdrive/.../Period 0/132831_1_frame_data.json" \
        --second-json "/data/share/SoccerTrack-v2/raw_bepro_from_gdrive/.../Period 0/132831_2_frame_data.json" \
        --metadata-json "/data/share/SoccerTrack-v2/data/raw/132831/132831_metadata.json" \
        --output-dir   "/data/share/SoccerTrack-v2/data/interim/132831"
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any

from loguru import logger


def load_metadata_json(metadata_json_path: Path | str) -> dict:
    """
    メタデータJSONを読み込み、そのままdictで返す。
    """
    metadata_json_path = Path(metadata_json_path)
    if not metadata_json_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_json_path}")

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Loaded metadata JSON from {metadata_json_path}")
    return meta


def build_team_mapping_from_metadata(meta: dict) -> Dict[str, str | None]:
    """
    メタデータJSONから player_id -> team_id のマッピングを作る。

    home_team / away_team の両方を見て、players 配列から引く。
    """
    mapping: Dict[str, str | None] = {}

    for side in ("home_team", "away_team"):
        team_info = meta.get(side)
        if not isinstance(team_info, dict):
            continue

        team_id = team_info.get("team_id")
        players = team_info.get("players", [])

        for pl in players:
            pid = pl.get("player_id")
            if pid is None:
                continue
            mapping[str(pid)] = team_id

    logger.info(f"Built player->teamId mapping for {len(mapping)} players")
    return mapping


def get_pitch_dimensions(meta: dict) -> tuple[float, float]:
    """
    ground_width, ground_height をメタデータJSONから取得。
    デフォルトは (105.0, 68.0)。
    """
    gw = meta.get("ground_width", 105.0)
    gh = meta.get("ground_height", 68.0)
    return float(gw), float(gh)


def load_tracking_json(json_path: Path | str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Bepro トラッキングJSONを読み込み、frame番号(int) -> オブジェクトリスト に変換。

    JSON構造:
        {
          "1": [ {...}, {...}, ... ],
          "2": [ {...}, ... ],
          ...
        }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Tracking JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    frame_dict: Dict[int, List[Dict[str, Any]]] = {}
    for k, v in raw.items():
        try:
            frame_idx = int(k)
        except ValueError:
            logger.warning(f"Invalid frame key '{k}' in {json_path}, skipping.")
            continue
        if not isinstance(v, list):
            logger.warning(f"Frame '{k}' value is not a list, skipping.")
            continue
        frame_dict[frame_idx] = v

    logger.info(f"Loaded {len(frame_dict)} frames from {json_path}")
    return frame_dict


def build_tracking_rows_for_half(
    frames: Dict[int, List[Dict[str, Any]]],
    team_mapping: Dict[str, str | None],
    event_period: str,
    ground_width: float,
    ground_height: float,
) -> List[Dict[str, Any]]:
    """
    あるハーフ( FIRST_HALF / SECOND_HALF )用の JSON から
    出力用 tracking row のリストを構築。

    - x,y は [0, ground_width],[0,ground_height] を [0,1] にスケーリング。
    - object が "PLAYER" のものは player_id を id とし、teamId をメタデータから付与。
    - object が "BALL"（あれば）は id="ball", teamId=None として扱う。
    """
    rows: List[Dict[str, Any]] = []

    for frame, obj_list in sorted(frames.items()):
        for obj in obj_list:
            obj_type = str(obj.get("object", "")).upper()
            match_time = obj.get("match_time", 0)
            x_m = obj.get("x", None)
            y_m = obj.get("y", None)

            if x_m is None or y_m is None:
                continue

            try:
                x_m = float(x_m)
                y_m = float(y_m)
            except Exception:
                continue

            # 0〜ground_width, 0〜ground_height を 0〜1 に正規化
            if ground_width <= 0 or ground_height <= 0:
                # 念のためゼロ割回避
                x_norm = 0.0
                y_norm = 0.0
            else:
                x_norm = x_m / ground_width
                y_norm = y_m / ground_height

            if obj_type == "PLAYER":
                pid = str(obj.get("player_id"))
                team_id = team_mapping.get(pid)
                obj_id = pid
            elif obj_type == "BALL":
                obj_id = "ball"
                team_id = None
            else:
                # PLAYER / BALL 以外は無視（必要なら後で追加）
                continue

            rows.append(
                {
                    "frame": int(frame),
                    "match_time": float(match_time),
                    "event_period": event_period,
                    "ball_status": "INPLAY",  # JSONに無いので一律 INPLAY としておく
                    "id": obj_id,
                    "x": x_norm,
                    "y": y_norm,
                    "teamId": team_id,
                }
            )

    logger.info(f"Built {len(rows)} rows for {event_period}")
    return rows


def write_csv(tracking_data: List[Dict[str, Any]], output_csv: Path | str) -> None:
    """
    tracking_data を CSV に書き出す。
    カラム構成は convert_raw_to_pitch_plane と同一。
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["frame", "match_time", "event_period", "ball_status", "id", "x", "y", "teamId"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in tracking_data:
            writer.writerow(row)

    logger.info(f"Wrote {len(tracking_data)} rows to {output_csv}")


def convert_bepro_json_to_pitch_plane(
    match_id: str,
    first_json_path: Path | str,
    second_json_path: Path | str,
    metadata_json_path: Path | str,
    output_dir: Path | str,
) -> None:
    """
    Bepro JSON(前半/後半) + メタデータJSON を pitch_plane_coordinates_*.csv に変換。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # メタデータ読み込み: player_id -> team_id, ピッチサイズ
    meta = load_metadata_json(metadata_json_path)
    team_mapping = build_team_mapping_from_metadata(meta)
    ground_width, ground_height = get_pitch_dimensions(meta)
    logger.info(f"Pitch size from metadata: width={ground_width}, height={ground_height}")

    # 1st half
    first_frames = load_tracking_json(first_json_path)
    first_rows = build_tracking_rows_for_half(
        first_frames,
        team_mapping,
        event_period="FIRST_HALF",
        ground_width=ground_width,
        ground_height=ground_height,
    )
    out_first = output_dir / f"{match_id}_pitch_plane_coordinates_1st_half.csv"
    write_csv(first_rows, out_first)

    # 2nd half
    second_frames = load_tracking_json(second_json_path)
    second_rows = build_tracking_rows_for_half(
        second_frames,
        team_mapping,
        event_period="SECOND_HALF",
        ground_width=ground_width,
        ground_height=ground_height,
    )
    out_second = output_dir / f"{match_id}_pitch_plane_coordinates_2nd_half.csv"
    write_csv(second_rows, out_second)

    logger.info("Finished converting Bepro JSON (tracking + metadata) to pitch-plane CSV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bepro tracking JSON (first/second half) + metadata JSON to pitch plane CSV (0-1 normalized)."
    )
    parser.add_argument("--match-id", type=str, required=True, help="Match id, e.g., 132831")
    parser.add_argument("--first-json", type=Path, required=True, help="Path to first half tracking JSON")
    parser.add_argument("--second-json", type=Path, required=True, help="Path to second half tracking JSON")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        required=True,
        help="Path to metadata JSON (with home_team / away_team and ground_width/height)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory, e.g., /data/share/SoccerTrack-v2/data/interim/132831",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_bepro_json_to_pitch_plane(
        match_id=args.match_id,
        first_json_path=args.first_json,
        second_json_path=args.second_json,
        metadata_json_path=args.metadata_json,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

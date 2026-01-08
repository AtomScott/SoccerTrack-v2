#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beproのframe_data.json (Period 0/1) を STC用 tracker_box_data.xml 形式に変換するスクリプト。

入力想定:
  /data/share/SoccerTrack-v2/data/raw/tracking_data/{video_id}/Period 0/{video_id}_1_frame_data.json
  /data/share/SoccerTrack-v2/data/raw/tracking_data/{video_id}/Period 1/{video_id}_2_frame_data.json

出力:
  /data/share/SoccerTrack-v2/data/raw/{video_id}/{video_id}_tracker_box_data.xml

※前提・仮定
- Period 0 -> eventPeriod="FIRST_HALF"
- Period 1 -> eventPeriod="SECOND_HALF"
- 座標 (x, y) はメタデータJSONの ground_width / ground_height で正規化して loc="[x_norm, y_norm]" として出力
- JSON内に BALL 情報が無い場合は、ダミーのボール loc="[0.5, 0.5]" speed="NA" を出力
- frameNumber は全Periodを通して1からの連番（必要なら後で調整可）

python3 data/interim/convert_tracking_data_json_to_xml.py 132831
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


BASE_RAW = Path("/data/share/SoccerTrack-v2/data/raw")


def load_metadata(video_id: str) -> tuple[float, float]:
    """
    メタデータJSONから ground_width, ground_height を取得する。
    見つからない場合は (105, 68) を返す。
    """
    meta_path = BASE_RAW / video_id / f"{video_id}_metadata.json"
    if not meta_path.exists():
        # デフォルトの国際規格サイズ
        return 105.0, 68.0

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gw = float(meta.get("ground_width", 105.0))
    gh = float(meta.get("ground_height", 68.0))
    return gw, gh


def load_period_frame_data(video_id: str, period_index: int) -> dict:
    """
    Period N の frame_data.json を読み込む。
    例:
      Period 0 -> {video_id}_1_frame_data.json
      Period 1 -> {video_id}_2_frame_data.json
    """
    period_dir = BASE_RAW / "tracking_data" / video_id / f"Period {period_index}"
    # ファイル名ルール: {video_id}_{period_index+1}_frame_data.json
    json_path = period_dir / f"{video_id}_{period_index + 1}_frame_data.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Frame data JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data は {"1": [...], "2": [...], ...} の形式を想定
    return data


def to_norm_loc(x: float, y: float, gw: float, gh: float) -> str:
    """
    グラウンド幅・高さで正規化して loc="[x_norm, y_norm]" 形式の文字列を返す。
    """
    x_norm = x / gw if gw > 0 else 0.0
    y_norm = y / gh if gh > 0 else 0.0
    # 0〜1に軽くクリップ
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    return f"[{x_norm:.2f}, {y_norm:.2f}]"


def build_xml_for_video(video_id: str) -> ET.ElementTree:
    """
    1つの video_id について、Period 0/1 を読み込んで <data> ノードを構築する。
    """
    ground_width, ground_height = load_metadata(video_id)

    root = ET.Element("data")

    # Period index -> eventPeriod のマッピング
    period_event_map = {
        0: "FIRST_HALF",
        1: "SECOND_HALF",
        # 必要になれば 2: "EXTRA_FIRST_HALF" などを追加
    }

    global_frame_number = 1

    for period_idx, event_period in period_event_map.items():
        try:
            period_data = load_period_frame_data(video_id, period_idx)
        except FileNotFoundError:
            # 該当Periodのデータが無ければスキップ
            continue

        # JSONのキーは "1", "2", ... なので数値順にソート
        for frame_key in sorted(period_data.keys(), key=lambda k: int(k)):
            objs = period_data[frame_key]
            if not objs:
                continue

            # match_time はフレーム内で共通と仮定して先頭から取る
            match_time = int(objs[0].get("match_time", 0))

            frame_elem = ET.SubElement(
                root,
                "frame",
                attrib={
                    "matchTime": str(match_time),
                    "frameNumber": str(global_frame_number),
                    "eventPeriod": event_period,
                    # ボール状態は情報が無いので一旦 "INPLAY" 固定
                    "ballStatus": "INPLAY",
                },
            )

            ball_loc_str = None
            ball_speed_str = None

            for obj in objs:
                obj_type = obj.get("object")
                x = float(obj.get("x", 0.0))
                y = float(obj.get("y", 0.0))
                speed = obj.get("speed", "NA")

                loc_str = to_norm_loc(x, y, ground_width, ground_height)

                if obj_type == "PLAYER":
                    player_id = obj.get("player_id")
                    if player_id is None:
                        continue

                    ET.SubElement(
                        frame_elem,
                        "player",
                        attrib={
                            "playerId": str(player_id),
                            "loc": loc_str,
                            "speed": str(speed),
                        },
                    )
                elif obj_type == "BALL":
                    # BALL が JSON にある場合はそれを使う
                    ball_loc_str = loc_str
                    ball_speed_str = str(speed)

            # BALL が無い場合はダミーを置く（loc="[0.5, 0.5]", speed="NA"）
            if ball_loc_str is None:
                ball_loc_str = "[0.50, 0.50]"
                ball_speed_str = "NA"

            ET.SubElement(
                frame_elem,
                "ball",
                attrib={
                    "playerId": "ball",
                    "loc": ball_loc_str,
                    "speed": ball_speed_str,
                },
            )

            global_frame_number += 1

    return ET.ElementTree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bepro tracking frame_data.json to tracker_box_data.xml format."
    )
    parser.add_argument("video_id", type=str, help="Video (match) id, e.g., 132831")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_id = args.video_id

    tree = build_xml_for_video(video_id)

    out_dir = BASE_RAW / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}_tracker_box_data.xml"

    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    print(f"Saved tracker_box_data XML to: {out_path}")


if __name__ == "__main__":
    main()

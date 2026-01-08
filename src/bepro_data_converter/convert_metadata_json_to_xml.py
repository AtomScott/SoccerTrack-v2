#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bepro 形式風の tracker_box_metadata.xml を、
metadata.json から生成するスクリプト。

想定:
  入力: /data/share/SoccerTrack-v2/data/raw/{match_id}/{match_id}_metadata.json
  出力: /data/share/SoccerTrack-v2/data/raw/{match_id}/{match_id}_tracker_box_metadata.xml

使い方:
  python3 /data/share/SoccerTrack-v2/data/interim/convert_metadata_json_to_xml.py 132877

この XML は、主に grant_labels_bepro.py が参照している
<players><player .../></players> 部分を正しく作ることを目的とした
「互換 XML」です。
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Bepro-style metadata JSON to tracker_box_metadata.xml"
    )
    p.add_argument(
        "match_id",
        type=str,
        help="Match (video) id, e.g., 132877",
    )
    p.add_argument(
        "--in-json",
        type=Path,
        help=(
            "Input metadata JSON path "
            "(default: /data/share/SoccerTrack-v2/data/raw/{match_id}/{match_id}_metadata.json)"
        ),
    )
    p.add_argument(
        "--out-xml",
        type=Path,
        help=(
            "Output tracker_box_metadata.xml path "
            "(default: /data/share/SoccerTrack-v2/data/raw/{match_id}/{match_id}_tracker_box_metadata.xml)"
        ),
    )
    return p.parse_args()


def load_metadata_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_xml_tree(meta: dict) -> ET.ElementTree:
    """
    JSONメタデータから XML の root (<metadata>) を構成する。
    grant_labels_bepro.py が使うのは players 部分のみなので、
    そこを確実に作ることを主目的にする。
    """

    # ルート
    root = ET.Element("metadata")

    # ---- match 要素（簡易版） ----
    match_elem = ET.SubElement(root, "match")
    match_elem.set("matchId", str(meta.get("match_id", "")))
    match_elem.set("matchTitle", str(meta.get("match_title", "")))
    match_elem.set("matchDatetime", str(meta.get("match_datetime", "")))
    # タイムゾーン等は JSON に無いので簡易に埋める
    match_elem.set("matchDatetimeLocal", str(meta.get("match_datetime", "")))
    match_elem.set("timezone", "Asia/Tokyo")
    match_elem.set("matchFullTime", str(meta.get("match_full_time", "")))
    match_elem.set("matchExtraTime", str(meta.get("match_extra_time", "")))

    # ---- pitch 要素 ----
    pitch_elem = ET.SubElement(root, "pitch")
    pitch_elem.set("width", str(meta.get("ground_width", 105)))
    pitch_elem.set("height", str(meta.get("ground_height", 68)))

    # ---- teams 要素（home/away） ----
    teams_elem = ET.SubElement(root, "teams")

    home = meta.get("home_team", {}) or {}
    away = meta.get("away_team", {}) or {}

    home_team_elem = ET.SubElement(teams_elem, "team")
    home_team_elem.set("id", str(home.get("team_id", "")))
    home_team_elem.set("name", str(home.get("team_name", "")))
    home_team_elem.set("nameEn", str(home.get("team_name", "")))
    home_team_elem.set("side", "home")

    away_team_elem = ET.SubElement(teams_elem, "team")
    away_team_elem.set("id", str(away.get("team_id", "")))
    away_team_elem.set("name", str(away.get("team_name", "")))
    away_team_elem.set("nameEn", str(away.get("team_name", "")))
    away_team_elem.set("side", "away")

    # ---- players 要素 ----
    players_elem = ET.SubElement(root, "players")

    def add_players(team_info: dict):
        team_id = str(team_info.get("team_id", ""))
        team_name = str(team_info.get("team_name", ""))
        for pl in team_info.get("players", []) or []:
            # JSON 側のキーを XML 側にマッピング
            pid = str(pl.get("player_id", ""))
            full_name = str(pl.get("full_name", ""))
            shirt_number = str(pl.get("shirt_number", ""))
            position = str(pl.get("initial_position_name", ""))  # ex: "GK", "CM", ...

            player_elem = ET.SubElement(players_elem, "player")
            player_elem.set("id", pid)
            player_elem.set("name", full_name)
            player_elem.set("nameEn", full_name)
            player_elem.set("shirtNumber", shirt_number)
            player_elem.set("position", position)
            # team 情報
            player_elem.set("teamId", str(pl.get("team_id", team_id)))
            player_elem.set("teamName", team_name)
            player_elem.set("teamNameEn", team_name)

    add_players(home)
    add_players(away)

    # activePlayers や period は今回は必須ではないので省略
    # （必要になれば JSONに応じて生成ロジックを拡張）

    tree = ET.ElementTree(root)
    return tree


def save_xml(tree: ET.ElementTree, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Python 3.9+ なら indent で整形出力できる
    try:
        ET.indent(tree, space="  ", level=0)
    except Exception:
        pass

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved XML: {output_path}")


def main():
    args = parse_args()
    match_id = args.match_id

    # デフォルトパス設定
    default_json = Path(f"/data/share/SoccerTrack-v2/data/raw/{match_id}/{match_id}_metadata.json")
    default_xml = Path(f"/data/share/SoccerTrack-v2/data/raw/{match_id}/{match_id}_tracker_box_metadata.xml")

    in_json = args.in_json or default_json
    out_xml = args.out_xml or default_xml

    if not in_json.exists():
        raise FileNotFoundError(f"Input metadata JSON not found: {in_json}")

    meta = load_metadata_json(in_json)
    tree = build_xml_tree(meta)
    save_xml(tree, out_xml)


if __name__ == "__main__":
    main()

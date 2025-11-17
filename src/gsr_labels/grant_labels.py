#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOT形式のGT CSVにヘッダとメタ情報を付与するスクリプト。
- supercategory を "object" で追加
- role を position(GKならgoalkeeper, それ以外はplayer) から決定
- category_id を role に応じて (goalkeeper=2, player=1)
- jersey を shirtNumber から文字列で付与
- team を left/right のチームID判定で付与
- class_name 列は削除
- 出力ヘッダ順は指定順

使い方:
    python3 grant_labels.py 117093 left right
"""

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment MOT CSV with headers and metadata columns.")
    p.add_argument("video_id", type=str, help="Video (match) id, e.g., 117093")
    p.add_argument("left_team", type=str, help="Left teamId, e.g., 9701")
    p.add_argument("right_team", type=str, help="Right teamId, e.g., 9834")
    return p.parse_args()


def load_player_metadata(xml_path: Path) -> dict:
    """
    XMLから playerId -> {position, shirtNumber(str), teamId(str)} の辞書を作成
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    players = {}
    for pl in root.findall(".//players/player"):
        pid = pl.attrib.get("id")
        if pid is None:
            continue
        players[pid] = {
            "position": pl.attrib.get("position", ""),
            "shirtNumber": str(pl.attrib.get("shirtNumber", "") or ""),
            "teamId": pl.attrib.get("teamId", ""),
        }
    return players


def map_role_and_cat(position: str) -> tuple[str, int]:
    """
    position が 'GK' のとき role=goalkeeper, category_id=2
    それ以外は role=player, category_id=1
    """
    if (position or "").upper() == "GK":
        return "goalkeeper", 2
    return "player", 1


def team_side_for(team_id: str, left_team: str, right_team: str) -> str:
    """
    teamId を left/right/unknown にマップ
    """
    if str(team_id) == str(left_team):
        return "left"
    if str(team_id) == str(right_team):
        return "right"
    return "unknown"


def main():
    args = parse_args()
    video_id = args.video_id
    left_team = args.left_team
    right_team = args.right_team

    # 入出力パス
    base_dir = Path(f"/data/share/SoccerTrack-v2/data/interim/{video_id}")
    in_csv = base_dir / f"{video_id}_ground_truth_mot_2nd_half_distorted.csv"
    out_csv = base_dir / f"{video_id}_player_gsr_2nd_half_distorted.csv"
    meta_xml = Path(f"/data/share/SoccerTrack-v2/data/raw/{video_id}/{video_id}_tracker_box_metadata.xml")

    # 存在チェック
    if not in_csv.exists():
        print(f"ERROR: input csv not found: {in_csv}", file=sys.stderr)
        sys.exit(1)
    if not meta_xml.exists():
        print(f"ERROR: metadata xml not found: {meta_xml}", file=sys.stderr)
        sys.exit(1)

    # メタデータ読込
    player_meta = load_player_metadata(meta_xml)

    # 入力CSVはヘッダ無しの11列: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z,class_name
    cols_in = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z", "class_name"]
    df = pd.read_csv(in_csv, header=None, names=cols_in)

    # id を str でも int でも照合できるよう揃える（XMLは文字列ID）
    # 行側の id は数値のことが多いので文字列化
    df["id_str"] = df["id"].astype(str)

    # supercategory 追加（全て "object"）
    df["supercategory"] = "object"

    # role, category_id, jersey, team を付与
    roles = []
    cats = []
    jerseys = []
    teams = []

    for pid in df["id_str"]:
        meta = player_meta.get(pid, None)
        if meta is None:
            # メタデータに無い場合のフォールバック
            role, cat = map_role_and_cat("")  # -> player, 1
            jersey = ""
            team_side = "unknown"
        else:
            role, cat = map_role_and_cat(meta.get("position", ""))
            jersey = meta.get("shirtNumber", "")
            team_side = team_side_for(meta.get("teamId", ""), left_team, right_team)

        roles.append(role)
        cats.append(cat)
        jerseys.append(jersey)
        teams.append(team_side)

    df["role"] = roles
    df["category_id"] = cats
    df["jersey"] = jerseys
    df["team"] = teams

    # 出力列の順序に並べ替え（class_name と id_str は落とす）
    cols_out = [
        "frame",
        "id",
        "supercategory",
        "category_id",
        "role",
        "jersey",
        "team",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "conf",
        "x",
        "y",
        "z",
    ]

    # class_name を削除、id_str も削除
    df = df.drop(columns=["class_name", "id_str"], errors="ignore")

    # 列の型を軽く整える（任意）
    int_cols = ["frame", "id", "category_id"]
    for c in int_cols:
        # 既にfloatなら小数点が入っていない前提で安全に変換
        try:
            df[c] = df[c].astype(int)
        except Exception:
            pass

    # 並べ替え＆保存（ヘッダ付き）
    df = df[cols_out]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
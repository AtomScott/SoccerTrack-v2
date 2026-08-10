"""
Script to overlay possessions on frames in a video based on specified annotations from a JSON file.

This script reads video data and corresponding event annotations in JSON format, identifies the frames
where events occur, and overlays possessions on these frames for a specified number of frames (default: 5 frames).
The output video is saved with the possessions applied, and processing is limited to the first 2 minutes of the video.

The annotations JSON follows the structure:
{
    "UrlLocal": "",
    "UrlYoutube": "",
    "annotations": [
        {
            "gameTime": "1 - mm:ss",
            "possession": "event_possession",
            "position": "frame_number",
            "team": "",
            "visibility": ""
        },
        ...
    ]
}
"""

import argparse
import cv2
import pandas as pd
import numpy as np
import json
from loguru import logger

def parse_arguments():
    """
    Parse command line arguments for match ID and event class count.

    Returns:
        argparse.Namespace: Parsed command line arguments containing match_id and num_class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_id', help="Match ID for the video and annotation files")
    return parser.parse_args()

def main():
    # Example usage
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    
    for match_id in match_ids:
        filtered_tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_filtered_pitch_plane_coordinates.csv'
        player_to_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_player_to_player.json'
        nearest_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_nearest_player_data.csv'
        output_video_path = f'data/interim/event_visualization/{match_id}/{match_id}_player_to_player.mp4'

        # ファイルを読み込み
        filtered_tracking_df = pd.read_csv(filtered_tracking_path)
        nearest_player_df = pd.read_csv(nearest_player_path)
        filtered_tracking_df.reset_index
        nearest_player_df.reset_index
        with open(player_to_player_path, 'r') as f:
            player_to_player_df = json.load(f)
        
        visualize_player_to_player(filtered_tracking_df, player_to_player_df, output_video_path)

def visualize_player_to_player(filtered_tracking_df: pd.DataFrame, player_to_player: pd.DataFrame, output_path: str) -> None:
    """
    Process a video and overlay event possessions and detections at specified frames.
    Also highlights the player closest to the ball in orange.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.
        possessions_df (pd.DataFrame): DataFrame containing event possessions.
        detections_df (pd.DataFrame): DataFrame containing detection data.
        output_path (str): Path to save the processed video.
    """
    # パラメータ設定
    video_duration_seconds = 180  # 動画の再生時間（秒）
    video_start_position = 10000
    fps = 25  # フレームレート
    frame_width, frame_height = 1150, 780  # サッカーコートの表示サイズ
    court_width, court_height = 105, 68  # 座標のサッカーコートの表示サイズ

    # サッカーコートの背景画像を生成（緑の長方形として簡易的に作成）
    court = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    court[:] = (0, 128, 0)
    soccer_court(court, frame_width, frame_height)

    # 動画作成用の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    possession_num = 0

    # フレームごとに選手とボールの位置を描画
    for frame_num in range(video_start_position, fps * video_duration_seconds + video_start_position):
        frame_data = filtered_tracking_df[filtered_tracking_df['match_time'] / 40.0 == frame_num]
        frame = court.copy()  # サッカーコートの背景をコピー

        for _, row in frame_data.iterrows():
            if frame_num <= 35:
                x, y = int(row['x'] * frame_width), int(row['y'] * frame_height)
            else:
                x, y = int(row['x'] * frame_width / court_width), int(row['y'] * frame_height / court_height)
            # ボールと選手の描画（ボールは赤、選手は青で描画）
            if row['id'] == 'ball':
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # ボールの位置
            else:
                color = (255, 0, 0) if row['teamId'] == 9701 else (0, 255, 0)  # チームごとに色分け
                cv2.circle(frame, (x, y), 5, color, -1)  # 選手の位置
                cv2.circle(frame, (x, y), int(1 * frame_width / court_width), color, 1) # 選手の円（Possession zone）

        # イベントラベルの表示処理（possessions_df）
        possession = player_to_player[possession_num]
        possession_str = possession['state']
        possession_start = possession['start']
        possession_end = possession['end']
        possession_start_position = float(possession_start['match_time'])
        possession_end_position = float(possession_end['match_time'])
        start_players = possession_start['players']
        start_player_id = start_players['player_id']
        # 40で四捨五入してポジションを計算
        corrected_possession_start_position = position_error_correction(possession_start_position)
        corrected_possession_end_position = position_error_correction(possession_end_position)
        if corrected_possession_start_position <= frame_num < corrected_possession_end_position:   
            display_possession(frame, possession_str, start_player_id, str(frame_num * 40), possession_type='possession')
        elif frame_num == corrected_possession_end_position:
            possession_num += 1

        out.write(frame)  # フレームを書き出し

        if frame_num % 1000 == 0:
            print(frame_num)
    out.release()  # 動画ファイルを保存
    print(f"Created file: {output_path}")

def soccer_court(court, frame_width, frame_height):
    # コートラインの色と太さを設定
    line_color = (255, 255, 255)  # 白
    line_thickness = 2
    # センターラインとセンターサークル
    cv2.line(court, (frame_width // 2, 0), (frame_width // 2, frame_height), line_color, line_thickness)
    cv2.circle(court, (frame_width // 2, frame_height // 2), 70, line_color, line_thickness)
    '''# ゴールエリア (左右のゴール付近)
    goal_area_width = 120
    goal_area_height = 440
    cv2.rectangle(court, (0, frame_height // 2 - goal_area_height // 2), (goal_area_width, frame_height // 2 + goal_area_height // 2), line_color, line_thickness)
    cv2.rectangle(court, (frame_width - goal_area_width, frame_height // 2 - goal_area_height // 2), (frame_width, frame_height // 2 + goal_area_height // 2), line_color, line_thickness)
    '''# ペナルティエリア
    penalty_area_width = 180
    penalty_area_height = 300
    cv2.rectangle(court, (0, frame_height // 2 - penalty_area_height // 2), (penalty_area_width, frame_height // 2 + penalty_area_height // 2), line_color, line_thickness)
    cv2.rectangle(court, (frame_width - penalty_area_width, frame_height // 2 - penalty_area_height // 2), (frame_width, frame_height // 2 + penalty_area_height // 2), line_color, line_thickness)
    # ゴール位置
    goal_width = 80
    goal_height = 30
    cv2.rectangle(court, (0, frame_height // 2 - goal_width // 2), (goal_height, frame_height // 2 + goal_width // 2), line_color, line_thickness)
    cv2.rectangle(court, (frame_width - goal_height, frame_height // 2 - goal_width // 2), (frame_width, frame_height // 2 + goal_width // 2), line_color, line_thickness)

def display_possession(frame, possession, player_id, frame_num, possession_type='possession'):
    """
    Display a possession or detection on a video frame.
    Args:
        frame (ndarray): The video frame to modify.
        possession (str): The text possession to overlay on the frame.
        possession_type (str): The type of possession ('possession' or 'detection').
    """
    if possession_type == 'possession':
        state_position = (780, 600)  # ラベルの表示位置
        player_id_position = (780, 700)  # ラベルの表示位置
        color = (128, 0, 0)
        frame_num_position = (780, 500)
        cv2.putText(frame, frame_num, frame_num_position, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
    else:
        return
    cv2.putText(frame, possession, state_position, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
    cv2.putText(frame, player_id, player_id_position, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)

def position_error_correction(position):
    remainder = position % 40
    # remainderが20以下か20以上かで分岐
    if remainder <= 20:
        corrected_position = (position - remainder) / 40.0
    else:
        corrected_position = (position - remainder) / 40.0 + 1
    return corrected_position

if __name__ == '__main__':
    main()

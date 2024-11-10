"""
Script to overlay labels on frames in a video based on specified annotations from a JSON file.

This script reads video data and corresponding event annotations in JSON format, identifies the frames
where events occur, and overlays labels on these frames for a specified number of frames (default: 5 frames).
The output video is saved with the labels applied, and processing is limited to the first 2 minutes of the video.

The annotations JSON follows the structure:
{
    "UrlLocal": "",
    "UrlYoutube": "",
    "annotations": [
        {
            "gameTime": "1 - mm:ss",
            "label": "event_label",
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
    parser.add_argument('--num_class', help="Number of event classes, e.g., '12' or '14'")
    parser.add_argument('--event_class', help="Kind of event classes, e.g., 'pass' or 'in_play'")
    return parser.parse_args()

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

def display_label(frame, label):
    """
    Display a label on a video frame.

    Args:
        frame (ndarray): The video frame to modify.
        label (str): The text label to overlay on the frame.
    """
    cv2.putText(frame, label, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (128, 0, 0), 1)

def visualize_event_tracking(tracking_df: pd.DataFrame, events_df: pd.DataFrame, nearest_players_df: pd.DataFrame, output_path: str) -> None:
    """
    Process a video and overlay event labels at specified frames based on annotations.
    Also highlights the player closest to the ball in orange.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.
        events_df (pd.DataFrame): DataFrame containing event annotations.
        nearest_players_df (pd.DataFrame): DataFrame containing nearest player data.
        output_path (str): Path to save the processed video.
    """
    # パラメータ設定
    video_duration_seconds = 120  # 動画の再生時間（秒）
    fps = 25  # フレームレート
    frame_width, frame_height = 1050, 680  # サッカーコートの表示サイズ

    # サッカーコートの背景画像を生成（緑の長方形として簡易的に作成）
    court = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    court[:] = (0, 128, 0)
    soccer_court(court, frame_width, frame_height)

    # 動画作成用の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    annotations = events_df['annotations']
    annotation_num = 0
    last_annotation_frame = 0

    # フレームごとに選手とボールの位置を描画
    for frame_num in range(fps * video_duration_seconds):
        frame_data = tracking_df[tracking_df['match_time'] / 40.0 == frame_num]
        frame = court.copy()  # サッカーコートの背景をコピー
        # ボールに最も近い選手の情報を取得
        nearest_info = nearest_players_df[nearest_players_df.index == frame_num]
        print(frame_num, nearest_players_df[nearest_players_df.index == frame_num])
        nearest_player_id = nearest_info['nearest_player'].values[0] if not nearest_info.empty else None

        for _, row in frame_data.iterrows():
            x, y = int(row['x'] * frame_width), int(row['y'] * frame_height)
            # ボールと選手の描画（ボールは赤、選手は青で描画）
            if row['id'] == 'ball':
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # ボールの位置
            else:
                color = (255, 0, 0) if row['teamId'] == 9701 else (0, 255, 0)  # チームごとに色分け
                cv2.circle(frame, (x, y), 5, color, -1)  # 選手の位置
                # 最も近い選手をオレンジで囲む
                if row['id'] == nearest_player_id:
                    cv2.circle(frame, (x, y), 15, (0, 165, 255), 3)  # オレンジ色で囲む
        # イベントラベルの表示処理
        annotation = annotations[annotation_num]
        position = float(annotation['position'])
        remainder = position % 40
        if remainder <= 20:
            corrected_position = (position - remainder) / 40.0
        else:
            corrected_position = (position - remainder) / 40.0 + 1
        position = corrected_position
        
        if frame_num == position:
            label = annotation['label']
            display_label(frame, label)
            annotation_num += 1
            last_annotation_frame = frame_num + 15
        elif frame_num < last_annotation_frame:
            display_label(frame, label)
        out.write(frame)  # フレームを書き出し

        if frame_num % 100 == 0:
            print(frame_num)
    out.release()  # 動画ファイルを保存
    print(f"Created file: {output_path}")

if __name__ == '__main__':
    # Example usage
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    num_class = str(args.num_class)
    event_class = str(args.event_class)
    
    for match_id in match_ids:
        tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_pitch_plane_coordinates.csv' # pitch_plane_coordinates, ball_position
        event_path = f'data/raw/{match_id}/{match_id}_{num_class}_class_events.json'
        detection_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_{event_class}_detection.json'
        nearest_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_nearest_player_data.csv'
        output_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_{event_class}_detection.json'
        output_video_path = f'data/interim/event_visualization/{match_id}/{match_id}_nearest_player.mp4'

        # ファイルを読み込み
        tracking_df = pd.read_csv(tracking_path)
        nearest_player_df = pd.read_csv(nearest_player_path)
        tracking_df.reset_index
        nearest_player_df.reset_index
        with open(detection_json_path, 'r') as f: # detection_json_path or event_path
            events_df = json.load(f)
        
        visualize_event_tracking(tracking_df, events_df, nearest_player_df, output_video_path)

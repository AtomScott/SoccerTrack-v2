"""
Script to detect events in a soccer match video from tracking data in a CSV file.

This script reads tracking data of the ball and players, 
to identify events based on changes in ball movement and players position. 

The tracking data CSV structure is as follows:
frame,match_time,event_period,ball_status,id,x,y,teamId
"""

import argparse
import cv2
import pandas as pd
import numpy as np
import json
from loguru import logger
import os

def parse_arguments():
    """
    Parse command line arguments for match ID and event class count.

    Returns:
        argparse.Namespace: Parsed command line arguments containing match_id and num_class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_id', help="Match ID for the video and annotation files")
    parser.add_argument('--num_class', help="Number of event classes, e.g., '12' or '14'")
    return parser.parse_args()

def main():
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    num_class = str(args.num_class)
    
    for match_id in match_ids:
        tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_pitch_plane_coordinates.csv'
        event_path = f'data/raw/{match_id}/{match_id}_{num_class}_class_events.json'
        output_video_path = f'data/interim/event_visualization/{match_id}/{match_id}_event_tracking.mp4'
        output_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_in_play_detection.json'
        # ファイルを読み込み
        tracking_df = pd.read_csv(tracking_path)
        with open(event_path, 'r') as f:
            events_df = json.load(f)
        in_play_detection(tracking_df, events_df, output_video_path, output_json_path)

def in_play_detection(tracking_df, events_df, output_video_path, output_json_path):
    """
    Process tracking data to detect events and overlay events on video.

    Args:
        tracking_df (pd.DataFrame): Tracking data for players and ball positions.
        events_df (dict): Event annotations.
        output_video_path (str): Path to save the processed video.
        output_json_path (str): Path to save pass detection results in JSON format.
    """
    results = detect(tracking_df, output_json_path)

def detect(tracking_df, output_json_path, sampling_rate=25, speed_threshold=0.05, min_stationary_duration=50):
    """
    First, detect to keep ball stopping more than 2 seconds. 
    Second, we put the label of end in play the time when stating to stop.
    Third, we detect whether ball is in the pitch.

    Args:
        tracking_df (pd.DataFrame): Tracking data containing ball and player positions.
        output_json_path (str): Path to save the detected events in JSON format.
        sampling_rate (int): Frames per second of the video (default: 25 fps).
        speed_threshold (float): Minimum speed (m/s) to consider the ball as moving (default: 0.5 m/s).
        min_stationary_duration (int): Minimum number of frames for the ball to be considered stationary (default: 50 frames).

    Returns:
        dict: A dictionary with annotations in the specified format.
    """
    # ボールのデータのみを抽出
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)

    # 停止状態を判定するためのフラグとカウンター
    stationary_start_frame = None
    stationary_start_time = None
    is_in_pitch = True
    end_in_play_detected = False  # 「end in play」ラベル付与を確認するフラグ
    outputs = []

    for i, row in ball_data.iterrows():
        frame = row['frame']
        match_time = row['match_time']
        if i == len(ball_data):
            continue

        # ボールが停止状態かの判定
        # まだ止まっていない
        if stationary_start_frame is None:
            # 止まったかも
            if ball_data.loc[i, 'x'] == ball_data.loc[i + 1, 'x'] and ball_data.loc[i, 'y'] == ball_data.loc[i + 1, 'y']:
                # 停止が開始した瞬間を記録
                stationary_start_frame = frame
                stationary_start_time = match_time
                stationary_stop_x = ball_data.loc[i, 'x']
                stationary_stop_y = ball_data.loc[i, 'y']
                # ピッチの内外を判定
                is_in_pitch = 0 <= ball_data.loc[i, 'x'] <= 1 and 0 <= ball_data.loc[i, 'y'] <= 1
        # 止まり始めている
        else:
            # 止まり続ける
            if ball_data.loc[i, 'x'] == stationary_stop_x and ball_data.loc[i, 'y'] == stationary_stop_y:
                # 一定時間（min_stationary_durationフレーム）以上停止している場合
                if (frame - stationary_start_frame) >= min_stationary_duration:
                    # 停止状態が2秒以上経過している場合
                    if not any(d['position'] == stationary_start_time for d in outputs):
                        label = "end in play (in)" if is_in_pitch else "end in play (out)"
                        outputs.append({
                            "gameTime": format_game_time(stationary_start_time),
                            "label": label,
                            "position": str(stationary_start_time),
                            "team": "",
                            "visibility": ""
                        })
                    end_in_play_detected = True  # 「end in play」が検出されたことを記録
            # 動き出す
            else:
                # 一定時間経ってから動き出した場合
                if end_in_play_detected:
                    # 直前に「end in play」が検出された場合のみ、「start in play」を付与
                    label = "start in play (in)" if is_in_pitch else "start in play (out)"
                    outputs.append({
                        "gameTime": format_game_time(match_time),
                        "label": label,
                        "position": str(match_time),
                        "team": "",
                        "visibility": ""
                    })
                # 一定時間経たずに動き出した場合も
                # 停止状態のリセット
                stationary_start_frame = None
                stationary_start_time = None
                end_in_play_detected = False  # フラグをリセット
        
        if i % 1000 == 0:
            print(i)

    # 結果をJSON形式で保存
    recognition_results = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "annotations": outputs
    }

    with open(output_json_path, 'w') as f:
        json.dump(recognition_results, f, indent=4)

def format_game_time(time):
    """
    時間を '1 - MM:SS' の形式でフォーマットするヘルパー関数。
    """
    seconds = time / 1000
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"1 - {minutes}:{seconds:02}"

def generate_recognition_results(pass_start_times, output_json_path, fps=25):
    """
    Generate recognition results in the specified JSON format for detected "PASS" events and save to a JSON file.

    Args:
        pass_start_frames (list of int): List of frame numbers where "PASS" events were detected.
        output_json_path (str): Path to save the JSON file.
        fps (int): Frames per second of the video, default is 25.

    Returns:
        str: Path to the saved JSON file.
    """
    outputs = []
    for time in pass_start_times:
        seconds = time / 1000
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        outputs.append({
            "gameTime": f"1 - {minutes}:{seconds:02}",
            "label": "PASS",
            "position": str(time),
            "team": "",
            "visibility": ""
        })

    recognition_results = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "annotations": outputs
    }

    with open(output_json_path, 'w') as f:
        json.dump(recognition_results, f, indent=4)

def evaluate_pass_accuracy(ground_truth, recognition_results, tolerance=5):
    """
    Evaluate the accuracy of "PASS" event detection by comparing recognition results to ground truth.

    Args:
        recognition_results (dict): Dictionary of recognized "PASS" events in JSON-like format.
        ground_truth (dict): Dictionary of ground truth "PASS" events in JSON-like format.
        tolerance (int): Tolerance in frames within which a "PASS" event is considered a match, default is 5.

    Returns:
        float: Accuracy as the ratio of correctly detected "PASS" events to ground truth "PASS" events.
    """
    recognized_pass_frames = [int(event["position"]) for event in recognition_results["annotations"] if event["label"] == "PASS"]
    ground_truth_pass_frames = [int(event["position"]) for event in ground_truth["annotations"] if event["label"] == "PASS"]

    correct_detections = 0
    for gt_frame in ground_truth_pass_frames:
        if any(abs(gt_frame - rec_frame) <= tolerance for rec_frame in recognized_pass_frames):
            correct_detections += 1

    accuracy = correct_detections / len(ground_truth_pass_frames) if ground_truth_pass_frames else 0
    return accuracy

if __name__ == '__main__':
    main()
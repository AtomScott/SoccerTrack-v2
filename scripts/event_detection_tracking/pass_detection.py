"""
Script to detect passes in a soccer match video from tracking data in a CSV file.

This script reads tracking data of the ball and players, 
to identify pass events based on changes in ball velocity and proximity to players. 
It calculates the ball's speed and acceleration and determines whether a pass has occurred 
based on acceleration thresholds and ball possession shifts between teams. 
Identified passes are then used to label specific frames in the video.

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
from pykalman import KalmanFilter


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
        output_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_pass_detection.json'
        output_csv_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_ball_position.csv'
        # ファイルを読み込み
        tracking_df = pd.read_csv(tracking_path)
        with open(event_path, 'r') as f:
            events_df = json.load(f)
        pass_detection(tracking_df, events_df, output_video_path, output_json_path, output_csv_path)

def pass_detection(tracking_df, events_df, output_video_path, output_json_path, output_csv_path):
    """
    Process tracking data to detect passes and overlay events on video.

    Args:
        tracking_df (pd.DataFrame): Tracking data for players and ball positions.
        events_df (dict): Event annotations.
        output_video_path (str): Path to save the processed video.
        output_json_path (str): Path to save pass detection results in JSON format.
    """
    # pass_start_times = detect_pass_with_player(tracking_df)
    # pass_start_times = detect_pass_with_velocity_vector(tracking_df, output_csv_path)
    a = detect_pass_with_velocity_vector(tracking_df, output_csv_path)

    # Example processing and overlay logic
    # generate_recognition_results(pass_start_times, output_json_path)

    with open(output_json_path, 'r') as f:
        outputs_df = json.load(f)

    # evaluate_pass_accuracy(events_df, outputs_df)

import numpy as np
import pandas as pd

def detect_pass_with_player(tracking_df, sampling_rate=25, window_size=5, min_hold_frames=10, pass_distance_threshold=0.1):
    """
    Detect pass start frames based on nearest player holding the ball and the ball's distance.

    Args:
        tracking_df (pd.DataFrame): Tracking data for players and ball positions.
        sampling_rate (int): Sampling rate of the data (default: 25 fps).
        window_size (int): Window size for moving average (default: 5).
        min_hold_frames (int): Minimum frames for a player to be considered holding the ball (default: 10).
        pass_distance_threshold (float): Distance threshold to detect a pass (default: 0.1).

    Returns:
        list: Frames where a pass start is detected.
    """
    pass_start_times = []
    ball_positions = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)
    # 一番近い選手と距離を取得
    near_team, near_dis, nearest_player = ball_dis(tracking_df)
    # 保持している選手を追跡
    current_holder = None
    hold_count = 0
    for i in range(1, len(ball_positions)):
        nearest_player_id = nearest_player[i]
        distance = near_dis[i]
        if nearest_player_id == current_holder:
            hold_count += 1
        else:
            current_holder = nearest_player_id
            hold_count = 1

        # 保持している選手が一定フレーム続いた場合
        if hold_count >= min_hold_frames:
            # ボールが一定距離以上離れた場合、パスを検出
            if distance > pass_distance_threshold:
                pass_start_times.append(ball_positions.loc[i, 'match_time'])
                hold_count = 0  # 検出後はリセット

    return pass_start_times

def detect_pass_with_velocity_vector(tracking_df, output_csv_path, sampling_rate=25, frame_diff=5, speed_change_threshold=2, direction_change_threshold=30, min_continuous_frames=3):
    """
    Detect pass start frames based on significant changes in ball velocity vector (speed and direction).

    Args:
        tracking_df (pd.DataFrame): Tracking data for players and ball positions.
        sampling_rate (int): Sampling rate of the data (default: 25 fps).
        frame_diff (int): Number of frames difference to calculate velocity vector (default: 5).
        speed_change_threshold (float): Threshold for detecting significant speed change (m/s).
        direction_change_threshold (float): Threshold for detecting significant direction change (degrees).
        min_continuous_frames (int): Minimum continuous frames exceeding threshold to detect a pass (default: 3).

    Returns:
        list: Frames where a pass start is detected.
    """
    ball_positions = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)
    ball_positions = kalman_smoothing(ball_positions)
    ball_positions = linear_interpolation(ball_positions)
    ball_positions.to_csv(output_csv_path, index=False)
    return 0
    velocity_vectors = np.zeros((len(ball_positions), 2))  # (vx, vy) for each frame
    speed = np.zeros(len(ball_positions))  # Speed (m/s)
    direction = np.zeros(len(ball_positions))  # Direction (degrees)
    pass_start_times = []
    count = 0

    # 速度ベクトルを計算
    for i in range(frame_diff, len(ball_positions)):
        delta_x = ball_positions.loc[i, 'x'] - ball_positions.loc[i - frame_diff, 'x']
        delta_y = ball_positions.loc[i, 'y'] - ball_positions.loc[i - frame_diff, 'y']
        vx = delta_x * (sampling_rate / frame_diff)
        vy = delta_y * (sampling_rate / frame_diff)
        velocity_vectors[i] = [vx, vy]
        speed[i] = np.sqrt(vx**2 + vy**2)
        direction[i] = np.degrees(np.arctan2(vy, vx))

    # 速度ベクトルの変化を検出
    for i in range(1, len(speed)):
        speed_change = abs(speed[i] - speed[i - 1])
        direction_change = abs(direction[i] - direction[i - 1])
        # 方向の変化を 180 度以下に正規化
        direction_change = min(direction_change, 360 - direction_change)
        print(speed_change, direction_change)

        if speed_change > speed_change_threshold or direction_change > direction_change_threshold:
            count += 1
            if count >= min_continuous_frames:
                pass_start_times.append(ball_positions.loc[i, 'match_time'])
                count = 0  # 検出後はリセット
        else:
            count = 0

    return pass_start_times

def kalman_smoothing(data):
    observations = data[['x', 'y']].to_numpy()
    kf = KalmanFilter(initial_state_mean=[observations[0, 0], observations[0, 1], 0, 0],
                      initial_state_covariance=np.eye(4),
                      transition_matrices=[[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                      observation_matrices=[[1, 0, 0, 0], [0, 1, 0, 0]],
                      observation_covariance=np.eye(2) * 0.1,
                      transition_covariance=np.eye(4) * 0.01)

    smoothed_state_means, _ = kf.smooth(observations)
    data['x_smooth'] = smoothed_state_means[:, 0]
    data['y_smooth'] = smoothed_state_means[:, 1]
    return data[['frame', 'match_time', 'x_smooth', 'y_smooth']]

def linear_interpolation(data, max_gap=10):
    data['x_smooth'] = data['x_smooth']
    data['y_smooth'] = data['y_smooth']

    for i in range(1, len(data) - 1):
        if abs(data.loc[i, 'x_smooth'] - data.loc[i - 1, 'x_smooth']) > max_gap:
            data.loc[i, 'x_smooth'] = (data.loc[i - 1, 'x_smooth'] + data.loc[i + 1, 'x_smooth']) / 2
        if abs(data.loc[i, 'y_smooth'] - data.loc[i - 1, 'y_smooth']) > max_gap:
            data.loc[i, 'y_smooth'] = (data.loc[i - 1, 'y_smooth'] + data.loc[i + 1, 'y_smooth']) / 2

    return data[['frame', 'match_time', 'x_smooth', 'y_smooth']]

def ball_dis(tracking_df, fps=25):
    """
    Determine the nearest player and the distance to the ball at each frame.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.
        fps (int): Frames per second of the video.

    Returns:
        tuple: Three numpy arrays: near_team, near_dis, and nearest_player.
               near_team: Indicates which team is nearest to the ball (0 for defense, 1 for attack).
               near_dis: Distance to the nearest player from the ball.
               nearest_player: ID of the nearest player to the ball.
    """
    # キャッシュファイルが存在する場合は読み込み
    output_csv_path="data/interim/pitch_plane_coordinates/117093/117093_nearest_player_data.csv"
    if os.path.exists(output_csv_path):
        print(f"Loading cached data from {output_csv_path}")
        nearest_data = pd.read_csv(output_csv_path)
        return nearest_data['near_team'],nearest_data['near_dis'],nearest_data['nearest_player']
    
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['frame', 'x', 'y']].reset_index(drop=True)
    player_data = tracking_df[tracking_df['id'] != 'ball'][['frame', 'id', 'teamId', 'x', 'y']]
    near_team = np.zeros(len(ball_data), dtype=int)
    near_dis = np.zeros(len(ball_data))
    nearest_player = np.zeros(len(ball_data), dtype=object)

    for idx, (frame, ball_x, ball_y) in enumerate(ball_data[['frame', 'x', 'y']].itertuples(index=False)):
        players_in_frame = player_data[player_data['frame'] == frame][['id', 'teamId', 'x', 'y']]
        if players_in_frame.empty:
            continue
        # プレイヤーとの距離を計算
        distances = np.sqrt((players_in_frame['x'] - ball_x)**2 + (players_in_frame['y'] - ball_y)**2)
        min_idx = distances.idxmin()
        # 一番近い選手の情報を取得
        nearest_player[idx] = players_in_frame.loc[min_idx, 'id']
        near_dis[idx] = distances[min_idx]
        near_team[idx] = 0 if players_in_frame.loc[min_idx, 'teamId'] == 9701 else 1
        if idx % 1000 == 0:
            print(idx)
    # データフレームにまとめる
    nearest_data = pd.DataFrame({
        'frame': ball_data['frame'],
        'near_team': near_team,
        'near_dis': near_dis,
        'nearest_player': nearest_player
    })

    # CSVに保存
    nearest_data.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

    return near_team, near_dis, nearest_player

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
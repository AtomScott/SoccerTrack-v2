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
from collections import defaultdict

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
        output_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_event_detection.json'
        # ファイルを読み込み
        tracking_df = pd.read_csv(tracking_path)
        with open(event_path, 'r') as f:
            events_df = json.load(f)
        set_piece_detection(tracking_df, events_df, output_video_path, output_json_path)

def set_piece_detection(tracking_df, events_df, output_video_path, output_json_path):
    """
    Process tracking data to detect events and overlay events on video.

    Args:
        tracking_df (pd.DataFrame): Tracking data for players and ball positions.
        events_df (dict): Event annotations.
        output_video_path (str): Path to save the processed video.
        output_json_path (str): Path to save pass detection results in JSON format.
    """
    detect(tracking_df, output_json_path)

def detect(tracking_df, output_json_path, sampling_rate=25, speed_threshold=0.05, min_stationary_duration=50):
    """
    Scripts to detect event excluding PASS and DRIVE.
    First, we detect OUT from ball positional data. 
    After detection of OUT, we detect CK, GK and THROW IN from the positions ball begins to stop.
    Second, we detect FOUL if ball keeps stopping more than 2 seconds in the pitch. 
    After detection of FOUL, we detect FK if ball begins to move.

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
    # 一番近い選手と距離を取得
    all_distances_df = ball_dis(tracking_df)

    # 停止状態を判定するためのフラグとカウンター
    stationary_stop_frame = None
    stationary_out_frame = None
    after_x_out_CK_stop_frame = None
    after_x_out_GK_stop_frame = None
    after_y_out_stop_frame = None
    stationary_goal_frame = None
    is_in_pitch = True
    ball_player_id = None
    ball_lose_frame = None
    ball_possess_threshold = 0.05
    outputs = []

    for i, row in ball_data.iterrows():
        frame = row['frame']
        match_time = row['match_time']
        distances_data = all_distances_df[all_distances_df['match_time'] == match_time].reset_index(drop=True)

        # PASS → CROSS, HIGH PASS, SHOT, HEADER
        # 誰もまだボールを保持してない
        for i in range(len(distances_data)):
            player_data = distances_data[i].reset_index(drop=True)
            if ball_player_id is None:
                # 誰か保持
                print(player_data, type(player_data))
                if int(player_data['distance_to_ball']) <= ball_possess_threshold:
                    ball_player_id = player_data['player_id']
                    ball_team_id = player_data['team_id']
            # 既に誰かがボールを保持
            elif player_data['player_id'] == ball_player_id:
                # ボールを保持し続ける
                if player_data['distance_to_ball'] <= ball_possess_threshold:
                    pass
                    '''# ボールを奪われる
                    if nearest_player[i] != ball_player_id:
                        ball_player_id = None'''
                # ボールを離す
                else:
                    # 初めて離した
                    if ball_lose_frame is None:
                        # ボールが離れたフレームを記録
                        ball_lose_frame = frame
                        ball_lose_time = match_time
                        ball_lose_x = ball_data.loc[i, 'x']
                        ball_lose_y = ball_data.loc[i, 'y']
                    # 既にボールが離れている
                    else:
                        for next_player_data in distances_data:
                            # 誰かのエリアに侵入，誰かが保持
                            if next_player_data['distance_to_ball'] <= ball_possess_threshold:
                                # また同じ選手（not PASS）
                                if next_player_data['player_id'] == ball_player_id:
                                    pass
                                # 味方選手 + 相手選手 (PASS)
                                else:
                                    label = 'PASS'
                                    outputs.append({
                                        "gameTime": format_game_time(ball_lose_time),
                                        "label": label,
                                        "position": str(ball_lose_time),
                                        "team": "",
                                        "visibility": ""
                                    })
                                    ball_lose_frame = None
                                    ball_player_id = None

        # Out → CK, GK, TIの判定
        # まだ外出てない
        if stationary_out_frame is None:
            # out
            if ball_data.loc[i, 'x'] < 0.0 or ball_data.loc[i, 'x'] > 1.0 or ball_data.loc[i, 'y'] < 0.0 or ball_data.loc[i, 'y'] > 1.0:
                # 外に出たフレームを記録
                stationary_out_frame = frame
                stationary_out_time = match_time
                stationary_out_x = ball_data.loc[i, 'x']
                stationary_out_y = ball_data.loc[i, 'y']
                # ラベル付け
                label = 'OUT'
                outputs.append({
                    "gameTime": format_game_time(stationary_out_time),
                    "label": label,
                    "position": str(stationary_out_time),
                    "team": "",
                    "visibility": ""
                })
        # 外に出てる
        else:
            # 縦にOUT(→THROW IN)
            if (stationary_out_y < 0.0 or stationary_out_y > 1.0) and (0.0 <= stationary_out_x <= 1.0):
                # 既にライン際で止まっている
                if after_y_out_stop_frame is None:
                    # 止まったかも
                    if ball_data.loc[i, 'y'] < 0.05 or ball_data.loc[i, 'y'] > 0.95:
                        # 停止が開始した瞬間を記録
                        after_y_out_stop_frame = frame
                # ライン際で止まり始めている
                else:
                    # ライン際で止まり続ける
                    if ball_data.loc[i, 'y'] < 0.05 or ball_data.loc[i, 'y'] > 0.95:
                        pass
                    # 動き出す
                    else:
                        # 一定時間（min_stationary_durationフレーム）以上停止している場合 # 停止状態が1秒以上経過している場合
                        if (frame - after_y_out_stop_frame) >= min_stationary_duration / 2:
                            # ラベル付け
                            label = 'THROW IN'
                            outputs.append({
                                "gameTime": format_game_time(match_time),
                                "label": label,
                                "position": str(match_time),
                                "team": "",
                                "visibility": ""
                            })
                        after_y_out_stop_frame = None
                        stationary_out_frame = None
            # 横にOUT(→ CK or GK)
            else:
                # 既にコーナー付近で止まっている
                if after_x_out_CK_stop_frame is None:
                    # 止まったかも
                    if (ball_data.loc[i, 'x'] <= 0.1 or ball_data.loc[i, 'x'] >= 0.9) and (ball_data.loc[i, 'y'] <= 0.1 or ball_data.loc[i, 'y'] >= 0.9):
                        # 停止が開始した瞬間を記録
                        after_x_out_CK_stop_frame = frame
                # コーナー付近で止まり始めている
                else:
                    # コーナー付近で止まり続ける
                    if (ball_data.loc[i, 'x'] <= 0.1 or ball_data.loc[i, 'x'] >= 0.9) and (ball_data.loc[i, 'y'] <= 0.1 or ball_data.loc[i, 'y'] >= 0.9):
                        pass
                    # 動き出す
                    else:
                        # 一定時間（min_stationary_durationフレーム）以上停止している場合 # 停止状態が1秒以上経過している場合
                        if (frame - after_x_out_CK_stop_frame) >= min_stationary_duration / 5:
                            # ラベル付け
                            label = 'CORNER KICK'
                            outputs.append({
                                "gameTime": format_game_time(match_time),
                                "label": label,
                                "position": str(match_time),
                                "team": "",
                                "visibility": ""
                            })
                        after_x_out_CK_stop_frame = None
                        stationary_out_frame = None
    
                # 既にGK付近で止まっている
                if after_x_out_GK_stop_frame is None:
                    # 止まったかも
                    if (0.3 <= ball_data.loc[i, 'x'] <= 0.6 or 0.94 <= ball_data.loc[i, 'x'] <= 0.97) and (0.35 <= ball_data.loc[i, 'y'] <= 0.65):
                        # 停止が開始した瞬間を記録
                        after_x_out_GK_stop_frame = frame
                # コーナー付近で止まり始めている
                else:
                    # コーナー付近で止まり続ける
                    if (0.3 <= ball_data.loc[i, 'x'] <= 0.6 or 0.94 <= ball_data.loc[i, 'x'] <= 0.97) and (0.35 <= ball_data.loc[i, 'y'] <= 0.65):
                        pass
                    # 動き出す
                    else:
                        # 一定時間（min_stationary_durationフレーム）以上停止している場合 # 停止状態が1秒以上経過している場合
                        if (frame - after_x_out_GK_stop_frame) >= min_stationary_duration:
                            # ラベル付け
                            label = 'GOAL KICK'
                            outputs.append({
                                "gameTime": format_game_time(match_time),
                                "label": label,
                                "position": str(match_time),
                                "team": "",
                                "visibility": ""
                            })
                        after_x_out_GK_stop_frame = None
                        stationary_out_frame = None

        # FOUL → FK の検出
        # ボールが停止状態かの判定
        # まだ止まっていない
        if stationary_stop_frame is None:
            # 止まったかも
            if ball_data.loc[i, 'x'] == ball_data.loc[i + 1, 'x'] and ball_data.loc[i, 'y'] == ball_data.loc[i + 1, 'y']:
                # ピッチの外の場合，OUT検出と重複するのを回避
                if 0 < ball_data.loc[i, 'x'] < 1 and 0 < ball_data.loc[i, 'y'] < 1:
                    # 停止が開始した瞬間を記録
                    stationary_stop_frame = frame
                    stationary_stop_time = match_time
                    stationary_stop_x = ball_data.loc[i, 'x']
                    stationary_stop_y = ball_data.loc[i, 'y']
        # 止まり始めている
        else:
            # 止まり続ける
            if ball_data.loc[i, 'x'] == stationary_stop_x and ball_data.loc[i, 'y'] == stationary_stop_y:
                pass
            # 動き出す
            else:
                # 一定時間（min_stationary_durationフレーム）以上停止している場合 # 停止状態が4秒以上経過している場合
                if (frame - stationary_stop_frame) >= min_stationary_duration * 2:
                    if is_in_pitch:
                        label = "FOUL"
                        outputs.append({
                            "gameTime": format_game_time(stationary_stop_time),
                            "label": label,
                            "position": str(stationary_stop_time),
                            "team": "",
                            "visibility": ""
                        })
                        label = "FREE KICK"
                        outputs.append({
                            "gameTime": format_game_time(match_time),
                            "label": label,
                            "position": str(match_time),
                            "team": "",
                            "visibility": ""
                        })
                # 一定時間経たずに動き出した場合も
                # 停止状態のリセット
                stationary_stop_frame = None
                stationary_stop_time = None

        # GOAL の検出
        # ゴールの幅を通ってない
        if stationary_goal_frame is None:
            # ゴールの幅を通過
            if (0.44 <= ball_data.loc[i, 'y'] <= 0.56) and (ball_data.loc[i, 'x'] <= 0.0 or ball_data.loc[i, 'x'] >= 1.0):
                # 通過した瞬間を記録
                stationary_goal_frame = frame
                stationary_goal_time = match_time
        # 既にゴールの幅を通過
        else:
            # 60秒以内にボールが(0.5,0.5)に
            if (frame - stationary_goal_frame) >= 60 * sampling_rate:
                stationary_goal_frame = None
            if (0.49 <= ball_data.loc[i, 'x'] <= 0.51) and (0.49 <= ball_data.loc[i, 'y'] <= 0.51):
                label = "GOAL"
                outputs.append({
                    "gameTime": format_game_time(stationary_goal_time),
                    "label": label,
                    "position": str(stationary_goal_time),
                    "team": "",
                    "visibility": ""
                })
                stationary_goal_frame = None

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

def ball_dis(tracking_df, fps=25):
    """
    Calculate the distance of all players from the ball at each match time and output in CSV.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.
        fps (int): Frames per second of the video.

    Returns:
        pd.DataFrame: DataFrame with distances of all players to the ball for each match time.
    """
    # キャッシュファイルが存在する場合は読み込み
    output_csv_path = "data/interim/pitch_plane_coordinates/117093/117093_all_players_distance_to_ball.csv"
    if os.path.exists(output_csv_path):
        print(f"Loading cached data from {output_csv_path}")
        all_distances_df = pd.read_csv(output_csv_path)
        return all_distances_df.reset_index(drop=True)
    
    # ボールと選手データを分離
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['match_time', 'x', 'y']].reset_index(drop=True)
    player_data = tracking_df[tracking_df['id'] != 'ball'][['match_time', 'id', 'teamId', 'x', 'y']]

    # 全選手の距離データを格納するリスト
    distances_list = []

    # 各match_timeごとに全選手のボールからの距離を計算
    for idx, (match_time, ball_x, ball_y) in enumerate(ball_data[['match_time', 'x', 'y']].itertuples(index=False)):
        # 同じmatch_timeの選手データを取得し、idを基に一意の選手だけを抽出
        players_in_time = player_data[player_data['match_time'] == match_time][['id', 'teamId', 'x', 'y']].drop_duplicates(subset=['id'])
        
        if players_in_time.empty:
            continue

        # 各プレイヤーとの距離を計算し、データをリストに追加
        for _, player_row in players_in_time.iterrows():
            player_id = player_row['id']
            team_id = player_row['teamId']
            player_x = player_row['x']
            player_y = player_row['y']
            distance = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)

            distances_list.append({
                'match_time': match_time,
                'player_id': player_id,
                'team_id': team_id,
                'distance_to_ball': distance
            })
        
        # 進捗を出力
        if idx % 1000 == 0:
            print(f"Processed match_time {match_time}")

    # DataFrameに変換
    all_distances_df = pd.DataFrame(distances_list)

    # CSVに保存
    all_distances_df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

    return all_distances_df.reset_index(drop=True)


if __name__ == '__main__':
    main()
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
import math

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
        output_ball_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_ball_player.json'
        output_player_to_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_player_to_player.json'
        # ファイルを読み込み
        tracking_df = pd.read_csv(tracking_path)
        with open(event_path, 'r') as f:
            events_df = json.load(f)
        detect(tracking_df, output_json_path, output_ball_player_path, output_player_to_player_path)

def detect(tracking_df, output_json_path, output_ball_player_path, output_player_to_player_path, frame_rate=25, min_possess_duration=5, ball_possess_threshold = 0.05, min_stationary_duration=50):
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

    # 停止状態を判定するためのフラグとカウンター
    stationary_stop_frame = None
    stationary_out_frame = None
    after_x_out_CK_stop_frame = None
    after_x_out_GK_stop_frame = None
    after_y_out_stop_frame = None
    stationary_goal_frame = None
    is_in_pitch = True
    outputs = []

    # PASS → CROSS, HIGH PASS, SHOT, HEADER
    print('start detection of (PASS → CROSS, HIGH PASS, SHOT, HEADER)')

    # ボールのデータのみを抽出
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)
    # 全選手のボールとの距離を取得
    all_distances_df = ball_dis(tracking_df)
    # ボール保持者
    ball_player_df = ball_player(tracking_df, all_distances_df, output_ball_player_path)
    # ボール保持者の移り変わりを記録
    player_to_player_df = player_to_player(ball_player_df, output_player_to_player_path, frame_rate=25, min_possess_duration=5)
    for i, row in player_to_player_df.iterrows():
        # ラベル付け
        label = 'PASS'
        outputs.append({
            "gameTime": format_game_time(row['match_time']),
            "label": label,
            "position": str(row['match_time']),
            "team": "",
            "confidence": "1.0"
        })
    
    print('start detection of (Out → CK, GK, TI), (FOUL → FK), GOAL')

    for i, row in ball_data.iterrows():
        frame = row['frame']
        match_time = row['match_time']
        distances_data = all_distances_df[all_distances_df['match_time'] == match_time].reset_index(drop=True)

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
                    "confidence": "1.0"
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
                                "confidence": "1.0"
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
                                "confidence": "1.0"
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
                                "confidence": "1.0"
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
                            "confidence": "1.0"
                        })
                        label = "FREE KICK"
                        outputs.append({
                            "gameTime": format_game_time(match_time),
                            "label": label,
                            "position": str(match_time),
                            "team": "",
                            "confidence": "1.0"
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
            if (frame - stationary_goal_frame) >= 60 * frame_rate:
                stationary_goal_frame = None
            if (0.49 <= ball_data.loc[i, 'x'] <= 0.51) and (0.49 <= ball_data.loc[i, 'y'] <= 0.51):
                label = "GOAL"
                outputs.append({
                    "gameTime": format_game_time(stationary_goal_time),
                    "label": label,
                    "position": str(stationary_goal_time),
                    "team": "",
                    "confidence": "1.0"
                })
                stationary_goal_frame = None

        if i % 10000 == 0:
            print(i)
    
    # Sort outputs by frame for chronological order
    outputs.sort(key=lambda x: int(x['position']))

    # 結果をJSON形式で保存
    recognition_results = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "predictions": outputs
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
                'x': player_x,
                'y': player_y,
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

def ball_player(tracking_df, all_distances_df, output_ball_player_path, ball_possess_threshold = 0.05):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_ball_player_path):
        print(f"Loading cached data from {output_ball_player_path}")
        outputs_ball_player = pd.read_json(output_ball_player_path)
        return outputs_ball_player.reset_index(drop=True)
    
    # ボールのデータのみを抽出
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)
    outputs_ball_player = []
    
    for i, row in ball_data.iterrows():
        frame = row['frame']
        match_time = row['match_time']
        distances_data = all_distances_df[all_distances_df['match_time'] == match_time].reset_index(drop=True)

        # 距離が閾値以下の選手をフィルタリング
        close_players = distances_data[distances_data['distance_to_ball'] <= ball_possess_threshold]

        if close_players.empty:
            # 該当する選手がいない場合
            outputs_ball_player.append({"match_time": match_time})
        else:
            # 該当する選手がいる場合
            players_info = close_players[['player_id', 'team_id', 'x', 'y', 'distance_to_ball']].to_dict(orient="records")
            outputs_ball_player.append({
                "match_time": match_time,
                "players": players_info
            })
    with open(output_ball_player_path, 'w') as f:
        json.dump(outputs_ball_player, f, indent=4)

    return outputs_ball_player

def player_to_player(ball_player_df, output_player_to_player_path, frame_rate, min_possess_duration):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_player_to_player_path):
        print(f"Loading cached data from {output_player_to_player_path}")
        outputs_player_to_player = pd.read_json(output_player_to_player_path)
        return outputs_player_to_player.reset_index(drop=True)
    
    outputs_player_to_player = []
    last_player_id = None
    last_team_id = None
    possess_start_time = None
    possess_end_time = None

    for i, row in ball_player_df.iterrows():
        match_time = row["match_time"]
        players = row["players"]
        
        # NaNチェック
        if players is None or (isinstance(players, float) and math.isnan(players)):
            players = []  # 空のリストに置き換え
        
        # ボールを保持している選手がいるか確認
        if players:
            current_player_id = players[0]["player_id"]
            current_team_id = players[0]["team_id"]
        else:
            current_player_id = None
            current_team_id = None

        # ボール保持の変化をチェック
        if current_player_id != last_player_id:
            # 前の選手が規定時間以上保持していたらPASSと記録
            if possess_start_time is not None and (possess_end_time - possess_start_time) >= (min_possess_duration / frame_rate):
                outputs_player_to_player.append({
                    "match_time": possess_end_time,
                    "player_id": last_player_id,
                    "team_id": last_team_id,
                    "x": possess_end_x,
                    "y": possess_end_y
                })

            # 新しい保持者の情報をリセット
            last_player_id = current_player_id
            last_team_id = current_team_id
            possess_start_time = match_time if current_player_id else None

        # ボール保持の終了時間を更新
        if current_player_id:
            possess_end_time = match_time
            possess_end_x = players[0]["x"]
            possess_end_y = players[0]["y"]

    # 最後の保持者をチェック
    if possess_start_time is not None and (possess_end_time - possess_start_time) >= (min_possess_duration / frame_rate):
        outputs_player_to_player.append({
            "match_time": possess_end_time,
            "player_id": last_player_id,
            "team_id": last_team_id,
            "x": possess_end_x,
            "y": possess_end_y
        })
    with open(output_player_to_player_path, 'w') as f:
        json.dump(outputs_player_to_player, f, indent=4)

    return outputs_player_to_player

if __name__ == '__main__':
    main()
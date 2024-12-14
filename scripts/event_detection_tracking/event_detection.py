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
        tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_filtered_pitch_plane_coordinates.csv'
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

    outputs = []

    # PASS → CROSS, HIGH PASS, SHOT, HEADER
    print('start detection of (PASS → CROSS, HIGH PASS, SHOT, HEADER)')

    # 全選手のボールとの距離を取得
    all_distances_df = ball_dis(tracking_df)
    # ボール保持者
    ball_player_df = ball_player(tracking_df, all_distances_df, output_ball_player_path)
    # ボール保持者の移り変わりを記録
    player_to_player_df = player_to_player(ball_player_df, output_player_to_player_path, frame_rate=25, min_possess_duration=5)

    # 閾値 (方向ベクトルの角度差) を設定: 単位は度 (例: 30度以下はラベル付けしない)
    ANGLE_THRESHOLD = 50
    # OUT の後か記録
    after_OUT = False
    # 前回のパスを記録
    previous_position = None

    x_OUT = False
    y_OUT = False

    # Event を記録
    for i in range(len(player_to_player_df) - 1):
        if i >= 1000:
            break
        row = player_to_player_df.loc[i]
        # 現在の選手IDを取得
        current_player_id = row['player_id']
        # 現在のパス開始位置
        current_x = row['x']
        current_y = row['y']
        # 現在のパス開始位置
        current_position = np.array([row['x'], row['y']])
        # 次のパス受け取った人の位置を参照
        next_row = player_to_player_df.loc[i + 1]
        # 次のパス開始位置
        next_position = np.array([next_row['x'], next_row['y']])
        # フィルタをかけた元データ
        # origin_row = tracking_df[tracking_df['match_time'] == row['match_time']][['frame', 'match_time', 'x', 'y']]

        # OUTに依存
        # after OUT
        if after_OUT:
            # y方向にOUT(→THROW IN)
            if y_OUT:
                # 時間経過しているか(5s)かつ
                if row['match_time'] - OUT_match_time >= 3000:
                    if current_y <= 0.5 or current_y >= 67.5:
                        # 次の人がピッチ内　かつ　違う人
                        if (next_row['x'] >= 0.0 and next_row['x'] <= 105.0 and next_row['y'] >= 0.0 and next_row['y'] <= 68.0) and (current_player_id != next_row['player_id']):
                            # ラベル付け
                            label = 'THROW IN'
                            outputs.append({
                                "gameTime": format_game_time(row['match_time']),
                                "label": label,
                                "position": str(row['match_time']),
                                "team": "",
                                "confidence": "0.5"
                            })
                            after_OUT = False
                            y_OUT = False
            # 横にOUT(→ CK or GK)
            elif x_OUT:
                # 時間経過しているか(5s)
                print(row['match_time'], current_x, current_y, row['match_time'] - OUT_match_time)
                if row['match_time'] - OUT_match_time >= 5000:
                    # CK を記録
                    if (current_x <= 2.0 or current_x >= 103.0) and (current_y <= 2.0 or current_y >= 66.0):
                        # 次の人がピッチ内　かつ　違う人
                        if (next_row['x'] >= 0.0 and next_row['x'] <= 105.0 and next_row['y'] >= 0.0 and next_row['y'] <= 68.0):
                            # ラベル付け
                            label = 'CORNER KICK'
                            outputs.append({
                                "gameTime": format_game_time(row['match_time']),
                                "label": label,
                                "position": str(row['match_time']),
                                "team": "",
                                "confidence": "0.5"
                            })
                            after_OUT = False
                            x_OUT = False
                    # GOAL KICK を記録
                    elif (5.0 <= current_x <= 6.0 or 99.0 <= current_x <= 100.0) and (24 <= current_y <= 42):
                        # 次の人がピッチ内　かつ　違う人
                        if (next_row['x'] >= 0.0 and next_row['x'] <= 105.0 and next_row['y'] >= 0.0 and next_row['y'] <= 68.0):
                            # ラベル付け
                            label = 'GOAL KICK'
                            outputs.append({
                                "gameTime": format_game_time(row['match_time']),
                                "label": label,
                                "position": str(row['match_time']),
                                "team": "",
                                "confidence": "0.5"
                            })
                            after_OUT = False
                            x_OUT = False
        
        # after OUT ではない，In play
        else:
            # OUT を記録
            if type(row['player_id']) == str:
                if (row['y'] < 0.0 or row['y'] > 68.0) and (0.0 <= row['x'] <= 105.0):
                    # 次の保持者の間はどこにいるか
                    if (next_row['y'] < 1.0 or next_row['y'] > 67.0):
                        after_OUT = True
                        y_OUT = True
                        # ラベル付け
                        label = 'OUT'
                        outputs.append({
                            "gameTime": format_game_time(row['match_time']),
                            "label": label,
                            "position": str(row['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                elif (row['x'] < 0.0 or row['x'] > 105.0) and (0.0 <= row['y'] <= 68.0):
                    after_OUT = True
                    x_OUT = True
                    # ラベル付け(横方向は安心)
                    label = 'OUT'
                    outputs.append({
                        "gameTime": format_game_time(row['match_time']),
                        "label": label,
                        "position": str(row['match_time']),
                        "team": "",
                        "confidence": "0.5"
                    })
                OUT_match_time = row['match_time']
            
            # PASS を記録
            else:

                # 現在の選手と次の選手が同じ
                if current_player_id == next_row['player_id']:
                    # DRIVEの検出
                    # ワンタッチパス対策，一回トラップしてパスするのに少なくとも 0.2s かかると予想
                    if next_row['match_time'] - row['match_time'] >= 200:
                        # ラベル付け
                        label = 'DRIVE'
                        outputs.append({
                            "gameTime": format_game_time(row['match_time']),
                            "label": label,
                            "position": str(row['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                # 違う選手（パスの可能性高い）
                else:
                    # default
                    label = 'PASS'

                    # 誤検出
                    # 前回のパスが存在する場合、方向を比較する（スルーや上空対策）
                    if previous_position is not None:
                        last_pass_vector = current_position - previous_position
                        current_pass_vector = next_position - current_position
                        # ベクトルの角度差を計算
                        angle_difference = get_angle_difference(last_pass_vector, current_pass_vector)
                        # 角度差が閾値以下の場合はスキップ
                        if angle_difference <= ANGLE_THRESHOLD:
                            continue
                    # パスをしたわけではない場合（ボールを奪われた時），PLAYER_SUCCESFUL_TACKLE

                    
                    # CROSSの検出
                    if (current_position[0] <= 43.35 or current_position[0] >= 61.65) and \
                    (current_position[1] <= 13.84 or current_position[1] >= 54.16) and \
                    (next_position[0] <= 11 or next_position[0] >= 94) and \
                    (24.84 <= current_position[1] <= 43.16):
                        label = 'CROSS'

                    # SHOTの検出
                    elif (current_position[0] <= 30.0 or current_position[0] >= 75.0) and \
                    (13.84 <= current_position[1] <= 54.16) and \
                    (next_position[0] <= 3.0 or next_position[0] >= 102.0) and \
                    (24.84 <= next_position[1] <= 43.16):
                        label = 'SHOT'

                    # HIGH PASSの検出
                    elif np.linalg.norm(next_position - current_position) >= 45.0 :
                        label = 'HIGH PASS'

                    # GOALの検出
                    # elif 

                    # ラベル付け
                    outputs.append({
                        "gameTime": format_game_time(row['match_time']),
                        "label": label,
                        "position": str(row['match_time']),
                        "team": "",
                        "confidence": "0.5"
                    })
        
        '''# OUTに依存しない
        # 現在の選手と次の選手が同じ
        if current_player_id == next_row['player_id']:
            # DRIVEの検出
            # ワンタッチパス対策，一回トラップしてパスするのに少なくとも 0.2s かかると予想
            if next_row['match_time'] - row['match_time'] >= 200:
                # ラベル付け
                label = 'DRIVE'
                outputs.append({
                    "gameTime": format_game_time(row['match_time']),
                    "label": label,
                    "position": str(row['match_time']),
                    "team": "",
                    "confidence": "0.5"
                })
        # 違う選手（パスの可能性高い）
        else:
            # default
            label = 'PASS'

            # 誤検出
            # 前回のパスが存在する場合、方向を比較する（スルーや上空対策）
            if previous_position is not None:
                last_pass_vector = current_position - previous_position
                current_pass_vector = next_position - current_position
                # ベクトルの角度差を計算
                angle_difference = get_angle_difference(last_pass_vector, current_pass_vector)
                # 角度差が閾値以下の場合はスキップ
                if angle_difference <= ANGLE_THRESHOLD:
                    continue
            # パスをしたわけではない場合（ボールを奪われた時），PLAYER_SUCCESFUL_TACKLE

            
            # CROSSの検出
            if (current_position[0] <= 43.35 or current_position[0] >= 61.65) and \
            (current_position[1] <= 13.84 or current_position[1] >= 54.16) and \
            (next_position[0] <= 11 or next_position[0] >= 94) and \
            (24.84 <= current_position[1] <= 43.16):
                label = 'CROSS'

            # SHOTの検出
            elif (current_position[0] <= 30.0 or current_position[0] >= 75.0) and \
            (13.84 <= current_position[1] <= 54.16) and \
            (next_position[0] <= 3.0 or next_position[0] >= 102.0) and \
            (24.84 <= next_position[1] <= 43.16):
                label = 'SHOT'

            # HIGH PASSの検出
            elif np.linalg.norm(next_position - current_position) >= 45.0 :
                label = 'HIGH PASS'

            # GOALの検出
            # elif 

            # ラベル付け
            outputs.append({
                "gameTime": format_game_time(row['match_time']),
                "label": label,
                "position": str(row['match_time']),
                "team": "",
                "confidence": "0.5"
            })'''

        # 前回のパスを記録
        previous_position = current_position
    
    # Sort outputs by frame for chronological order
    outputs.sort(key=lambda x: int(float(x['position'])))

    # 結果をJSON形式で保存
    recognition_results = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "predictions": outputs
    }

    with open(output_json_path, 'w') as f:
        json.dump(recognition_results, f, indent=4)
    
    print(f"ファイルが保存されました: {output_json_path}")

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

            distances_list.append({
                'match_time': match_time,
                'player_id': player_row['id'],
                'team_id': player_row['teamId'],
                'x': player_row['x'],
                'y': player_row['y'],
                'distance_to_ball': np.sqrt((player_row['x'] - ball_x)**2 + (player_row['y'] - ball_y)**2)
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

def ball_player(tracking_df, all_distances_df, output_ball_player_path, ball_possess_threshold = 1.00):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_ball_player_path):
        print(f"Loading cached data from {output_ball_player_path}")
        outputs_ball_player = pd.read_json(output_ball_player_path)
        return outputs_ball_player.reset_index(drop=True)
    
    # ボールのデータのみを抽出
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)
    outputs_ball_player = []
    
    # 前回のボール位置を記録
    last_ball_x = 52.5
    last_ball_y = 34
    # 前回のOUTの時間を記録
    # last_out_match_time = 0.0

    after_OUT = True
    
    for i, row in ball_data.iterrows():
        match_time = row['match_time']
        ball_x = row['x']
        ball_y = row['y']
        distances_data = all_distances_df[all_distances_df['match_time'] == match_time].reset_index(drop=True)

        # 距離が閾値以下の選手をフィルタリング
        close_players = distances_data[distances_data['distance_to_ball'] <= ball_possess_threshold]

        # OUTの記録
        if (ball_x < 0.0 or ball_x > 105.0 or ball_y < 0.0 or ball_y > 68.0) and (last_ball_x >= 0.0 and last_ball_x <= 105.0 and last_ball_y >= 0.0 and last_ball_y <= 68.0): # and (match_time - last_match_time > 1000)
            outputs_ball_player.append({
                "match_time": match_time,
                "players": 'OUT',
                "x": ball_x,
                "y": ball_y
            })
            after_OUT = True
            # last_out_match_time = match_time
        # 距離が閾値以下の選手の記録
        else:
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
        
        last_ball_x = ball_x
        last_ball_y = ball_y

    with open(output_ball_player_path, 'w') as f:
        json.dump(outputs_ball_player, f, indent=4)

    return outputs_ball_player.reset_index(drop=True)

def player_to_player(ball_player_df, output_player_to_player_path, frame_rate, min_possess_duration):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_player_to_player_path):
        print(f"Loading cached data from {output_player_to_player_path}")
        outputs_player_to_player = pd.read_json(output_player_to_player_path)
        return outputs_player_to_player.reset_index(drop=True)
    
    outputs_player_to_player = []
    last_frame_player_id = None
    last_frame_players = None
    last_possess_player_id = None
    # ボール保持フレーム数（１の時対策必要）
    num_last_ball_possess_frame = 0

    for i in range(len(ball_player_df)):
        row = ball_player_df.loc[i]
        match_time = row["match_time"]
        current_frame_players = row["players"]

        # NaNチェックとデータ型変換
        if isinstance(current_frame_players, float) or current_frame_players is None:
            current_frame_players = []  # 空のリストに置き換え
        
        # 一番最初は全員距離0
        if i <= 1:
            last_frame_players = None
            current_frame_player_id = None
            current_frame_team_id = None
            last_possess_player_id = None
            continue
        
        # ボールを保持している選手がいるか確認
        # OUT
        if current_frame_players == 'OUT':
            if last_frame_players != 'OUT':
                outputs_player_to_player.append({
                        "match_time": float(match_time),
                        "player_id": 'OUT',
                        "team_id": 'OUT',
                        "x": float(row["x"]),
                        "y": float(row["y"])
                    })
            current_frame_player_id = None
            current_frame_team_id = None
            last_possess_player_id = None
            num_last_ball_possess_frame = 0
        else:
            if current_frame_players:
                current_frame_player_id = current_frame_players[0]["player_id"]
                current_frame_team_id = current_frame_players[0]["team_id"]
                # 1フレーム前のボール保持者と違う
                if current_frame_player_id != last_frame_player_id:
                    # 前回のボール保持者と違う
                    if current_frame_player_id != last_possess_player_id:
                        if (last_possess_player_id != None) and (num_last_ball_possess_frame >= 2):
                            # 前回の保持者でパスをしたと記録
                            outputs_player_to_player.append({
                                "match_time": float(last_possess_match_time),
                                "player_id": float(last_possess_player_id),
                                "team_id": float(last_possess_team_id),
                                "x": float(last_possess_x),
                                "y": float(last_possess_y)
                            })
                        # 前回の保持者からパスを受け取った人を記録
                        outputs_player_to_player.append({
                            "match_time": float(match_time),
                            "player_id": float(current_frame_player_id),
                            "team_id": float(current_frame_team_id),
                            "x": float(current_frame_players[0]["x"]),
                            "y": float(current_frame_players[0]["y"])
                        })
                    num_last_ball_possess_frame = 0
                # 前回のボール保持者として記録
                last_possess_match_time = match_time
                last_possess_x = current_frame_players[0]["x"]
                last_possess_y = current_frame_players[0]["y"]
                last_possess_player_id = current_frame_player_id
                last_possess_team_id = current_frame_team_id
                num_last_ball_possess_frame += 1

            else:
                current_frame_player_id = None
                current_frame_team_id = None

        last_frame_player_id = current_frame_player_id
        last_frame_players = current_frame_players

    # DataFrame 形式の場合
    with open(output_player_to_player_path, 'w') as f:
        json.dump(outputs_player_to_player, f, indent=4)

    return pd.DataFrame(outputs_player_to_player)

# ベクトルの角度差を計算する関数
def get_angle_difference(vec1, vec2):
    """
    Calculate the angle (in degrees) between two vectors.
    Args:
        vec1 (np.array): First vector.
        vec2 (np.array): Second vector.
    Returns:
        float: Angle difference in degrees.
    """
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

if __name__ == '__main__':
    main()
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
    parser.add_argument('--num_class', help="Number of event classes, e.g., '8' or '12'")
    return parser.parse_args()

def main():
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    num_class = str(args.num_class)
    
    for match_id in match_ids:
        tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_filtered_pitch_plane_coordinates.csv'
        output_video_path = f'data/interim/event_visualization/{match_id}/{match_id}_event_tracking.mp4'
        output_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_{num_class}_class_events_detection.json'
        output_ball_dis_path = "data/interim/pitch_plane_coordinates/117093/117093_all_players_distance_to_ball.csv"
        output_ball_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_ball_player.json'
        output_possession_group_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_possession_group.json'
        output_player_to_player_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_player_to_player.json'
        output_PASS_DRIVE_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_PASS_DRIVE.json'
        # ファイルを読み込み
        tracking_df = pd.read_csv(tracking_path)
        detect(tracking_df, output_json_path, output_ball_dis_path, output_ball_player_path, output_possession_group_path, output_player_to_player_path, output_PASS_DRIVE_path, num_class)

def detect(tracking_df, output_json_path, output_ball_dis_path, output_ball_player_path, output_possession_group_path, output_player_to_player_path, output_PASS_DRIVE_path, num_class, frame_rate=25, min_possess_duration=5, ball_possess_threshold = 0.05, min_stationary_duration=50):
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

    # 全選手のボールとの距離を取得
    all_distances_df, plus_team_id = ball_dis(tracking_df, output_ball_dis_path)
    # ボール保持者
    ball_player_df = ball_player(tracking_df, all_distances_df, output_ball_player_path)
    # データのグループ化
    possession_group_list = group_player_data(ball_player_df, output_possession_group_path)
    # ボール保持者の移り変わりを記録
    player_to_player_df = player_to_player(possession_group_list, output_player_to_player_path, frame_rate=25, min_possess_duration=5)
    # PASS, DRIVE, OUTを最初に記録
    PASS_DRIVE_df = PASS_DRIVE_OUT(player_to_player_df, output_PASS_DRIVE_path)
    # 各イベントの認識
    if num_class == str(8):
        # (HEADER), CROSS, THROW IN, SHOT, (FREE KICK), GOAL, CORNER KICK, GOAL KICK
        event_detect_Playbox(PASS_DRIVE_df, plus_team_id, output_json_path)
    elif num_class == str(12):
        # PASS, DRIVE, (HEADER), HIGH PASS, (OUT), CROSS, THROW IN, SHOT, (Ball Player Block), (Player Successful Tackle), (FREE KICK), GOAL
        event_detect_SoccerNet(PASS_DRIVE_df, plus_team_id, output_json_path)

def event_detect_Playbox(PASS_DRIVE_df, plus_team_id, output_json_path):

    predictions = []
    skip_count = 0
    x_OUT = False
    y_OUT = False

    for i in range(len(PASS_DRIVE_df)):
        if skip_count > 0:
            skip_count -= 1
            continue

        entry = PASS_DRIVE_df.loc[i]
        old_label = entry["label"]
        start = entry["start"]
        end = entry["end"]
        start_players = start["players"][0]
        end_players = end["players"][0]
        start_x = start_players.get("x")
        start_y = start_players.get("y")
        end_x = end_players.get("x")
        end_y = end_players.get("y")
        start_player_id = start_players.get("player_id")
        end_player_id = end_players.get("player_id")
        start_team_id = start_players.get("team_id")
        end_team_id = end_players.get("team_id")
        current_pass_vector = np.array([start_x, start_y]) - np.array([end_x, end_y])
        current_pass_len = np.linalg.norm(np.array([start_x - end_x, start_y - end_y]))

        if old_label == "PASS":

            # Case 1: PASS with valid start and end player IDs
            if start_player_id != "OUT" and end_player_id != "OUT":
                # The label in this case can be HIGH PASS, SHOT, CROSS
                # team attacking to plus direction
                if start_team_id == plus_team_id:
                    # CROSS
                    if (start_x >= 61.65) and (start_y <= 13.84 or start_y >= 54.16) and (end_x >= 94) and (24.84 <= end_y <= 43.16):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'CROSS',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                    # SHOT
                    elif (75.0 <= start_x <= 102.0) and (13.84 <= start_y <= 54.16) and (end_x >= 102.0) and (20 <= end_y <= 49):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'SHOT',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                # team attacking to minus direction
                else:
                    # CROSS
                    if (start_x <= 43.35) and (start_y <= 13.84 or start_y >= 54.16) and (end_x <= 11) and (24.84 <= end_y <= 43.16):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'CROSS',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                    # SHOT
                    elif (3.0 <= start_x <= 30.0) and (13.84 <= start_y <= 54.16) and (end_x <= 3.0) and (20 <= end_y <= 49):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'SHOT',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                # HIGH PASS
                if current_pass_len > 30.0:
                    predictions.append({
                        "gameTime": format_game_time(start['match_time']),
                        "label": 'HIGH PASS',
                        "position": str(start['match_time']),
                        "team": "",
                        "confidence": "0.5"
                    })

            # Case 2: PASS where end player ID is OUT
            elif end_player_id == "OUT":

                # Finding subsequent THROW IN, GOAL KICK, or CORNER KICK
                # OUT of side line or goal line
                # OUT of side line
                if (0.0 <= end_x <= 105.0) and (end_y < 0.0 or end_y > 68.0):
                    y_OUT = True
                # OUT of goal line
                elif (end_x < 0.0 or end_x > 105.0) and (0.0 <= end_y <= 68.0):
                    x_OUT = True
                # Placeholder logic for detecting these labels
                next_events = PASS_DRIVE_df[i + 1:i + 6].reset_index(drop=True)
                for j in range(len(next_events)):
                    event = next_events.loc[j]
                    if event.get("label") == "PASS":
                        start = event.get("start")
                        end = event.get("end")
                        start_players = start["players"][0]
                        end_players = end["players"][0]
                        start_x = start_players.get("x")
                        start_y = start_players.get("y")
                        end_x = end_players.get("x")
                        end_y = end_players.get("y")
                        # remember first event data
                        if j == 0:
                            first_start = start
                        # OUT of side line
                        if y_OUT:
                            if (start_y < 0.1 or start_y > 67.9):
                                if 0.2 <= end_y <= 67.8:
                                    predictions.append({
                                        "gameTime": format_game_time(start['match_time']),
                                        "label": "THROW IN",
                                        "position": str(start['match_time']),
                                        "team": "",
                                        "confidence": "0.5"
                                    })
                                    y_OUT = False
                                    x_OUT = False
                                    skip_count = j + 1  # Skip processed events in outer loop
                                    break
                            # last event (This means it can't find event)
                            if j == 4:
                                predictions.append({
                                    "gameTime": format_game_time(first_start['match_time']),
                                    "label": "THROW IN",
                                    "position": str(start['match_time']),
                                    "team": "",
                                    "confidence": "0.5"
                                })
                                y_OUT = False
                                x_OUT = False
                        # OUT of goal line
                        elif x_OUT:
                            # GK
                            if (5.0 <= start_x <= 6.0 or 99.0 <= start_x <= 100.0) and (24.0 <= start_y <= 42.0):
                                if (5.0 <= end_x < 100.0):
                                    predictions.append({
                                        "gameTime": format_game_time(start['match_time']),
                                        "label": "GOAL KICK",
                                        "position": str(start['match_time']),
                                        "team": "",
                                        "confidence": "0.5"
                                    })
                                    y_OUT = False
                                    x_OUT = False
                                    skip_count = j + 1  # Skip processed events in outer loop
                                    break
                            # CK
                            if (start_x <= 1.5 and start_y <= 1.5) or (start_x >= 103.5 and start_y <= 1.5) or (start_x <= 1.5 and start_y >= 66.5) or (start_x >= 103.5 and start_y >= 66.5):
                                if (1.5 < end_x < 103.5) and (1.5 < end_y < 66.5):
                                    predictions.append({
                                        "gameTime": format_game_time(start['match_time']),
                                        "label": "CORNER KICK",
                                        "position": str(start['match_time']),
                                        "team": "",
                                        "confidence": "0.5"
                                    })
                                    y_OUT = False
                                    x_OUT = False
                                    skip_count = j + 1  # Skip processed events in outer loop
                                    break
                            # GOAL
                            if (-2.0 <= end_x <= 0.0 or 105.0 <= end_x <-107.0) and (30.34 < end_y < 37.66):
                                predictions.append({
                                    "gameTime": format_game_time(start['match_time']),
                                    "label": "GOAL",
                                    "position": str(start['match_time']),
                                    "team": "",
                                    "confidence": "0.5"
                                })
                                y_OUT = False
                                x_OUT = False
                                skip_count = j + 1  # Skip processed events in outer loop
                                break
                            # last event (This means it can't find event)
                            if j == 4:
                                predictions.append({
                                    "gameTime": format_game_time(first_start['match_time']),
                                    "label": "GOAL KICK",
                                    "position": str(first_start['match_time']),
                                    "team": "",
                                    "confidence": "0.5"
                                })
                                y_OUT = False
                                x_OUT = False

    # Sort outputs by frame for chronological order
    predictions.sort(key=lambda x: int(float(x['position'])))

    # 結果をJSON形式で保存
    recognition_results = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "predictions": predictions
    }

    with open(output_json_path, 'w') as f:
        json.dump(recognition_results, f, indent=4)
    
    print(f"Data saved to {output_json_path}")

def event_detect_SoccerNet(PASS_DRIVE_df, plus_team_id, output_json_path):

    predictions = []
    skip_count = 0
    x_OUT = False
    y_OUT = False

    for i in range(len(PASS_DRIVE_df)):
        if skip_count > 0:
            skip_count -= 1
            continue
        entry = PASS_DRIVE_df.loc[i]
        old_label = entry["label"]
        start = entry["start"]
        end = entry["end"]
        start_players = start["players"][0]
        end_players = end["players"][0]
        start_x = start_players.get("x")
        start_y = start_players.get("y")
        end_x = end_players.get("x")
        end_y = end_players.get("y")
        start_player_id = start_players.get("player_id")
        end_player_id = end_players.get("player_id")
        start_team_id = start_players.get("team_id")
        end_team_id = end_players.get("team_id")
        current_pass_vector = np.array([start_x, start_y]) - np.array([end_x, end_y])
        current_pass_len = np.linalg.norm(np.array([start_x - end_x, start_y - end_y]))

        if old_label == "PASS":

            # Case 1: PASS with valid start and end player IDs
            if start_player_id != "OUT" and end_player_id != "OUT":
                # The label in this case can be HIGH PASS, SHOT, CROSS
                # team attacking to plus direction
                if start_team_id == plus_team_id:
                    # CROSS
                    if (start_x >= 61.65) and (start_y <= 13.84 or start_y >= 54.16) and (end_x >= 94) and (24.84 <= end_y <= 43.16):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'CROSS',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                    # SHOT
                    elif (75.0 <= start_x <= 100.0) and (13.84 <= start_y <= 54.16) and (end_x >= 102.0) and (20 <= end_y <= 49):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'SHOT',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                # team attacking to minus direction
                else:
                    # CROSS
                    if (start_x <= 43.35) and (start_y <= 13.84 or start_y >= 54.16) and (end_x <= 11) and (24.84 <= end_y <= 43.16):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'CROSS',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                    # SHOT
                    elif (5.0 <= start_x <= 30.0) and (13.84 <= start_y <= 54.16) and (end_x <= 3.0) and (20 <= end_y <= 49):
                        predictions.append({
                            "gameTime": format_game_time(start['match_time']),
                            "label": 'SHOT',
                            "position": str(start['match_time']),
                            "team": "",
                            "confidence": "0.5"
                        })
                # HIGH PASS
                if current_pass_len > 30.0:
                    predictions.append({
                        "gameTime": format_game_time(start['match_time']),
                        "label": 'HIGH PASS',
                        "position": str(start['match_time']),
                        "team": "",
                        "confidence": "0.5"
                    })
                # PASS
                else:
                    label = 'PASS'
                predictions.append({
                    "gameTime": format_game_time(start['match_time']),
                    "label": label,
                    "position": str(start['match_time']),
                    "team": "",
                    "confidence": "0.5"
                })

            # Case 2: PASS where end player ID is OUT
            elif end_player_id == "OUT":
                OUT_team_id = start_team_id
                predictions.append({
                    "gameTime": format_game_time(start['match_time']),
                    "label": "PASS",
                    "position": str(start['match_time']),
                    "team": "",
                    "confidence": "0.5"
                })
                predictions.append({
                    "gameTime": format_game_time(end['match_time']),
                    "label": "OUT",
                    "position": str(end['match_time']),
                    "team": "",
                    "confidence": "0.5"
                })

                # Finding subsequent THROW IN, GOAL KICK, or CORNER KICK
                # OUT of side line or goal line
                # OUT of side line
                if (0.0 <= end_x <= 105.0) and (end_y < 0.0 or end_y > 68.0):
                    y_OUT = True
                # OUT of goal line
                elif (end_x < 0.0 or end_x > 105.0) and (0.0 <= end_y <= 68.0):
                    x_OUT = True
                # Placeholder logic for detecting these labels
                next_events = PASS_DRIVE_df[i + 1:i + 6].reset_index(drop=True)
                for j in range(len(next_events)):
                    event = next_events.loc[j]
                    if event.get("label") == "PASS":
                        start = event.get("start")
                        end = event.get("end")
                        start_players = start["players"][0]
                        end_players = end["players"][0]
                        start_x = start_players.get("x")
                        start_y = start_players.get("y")
                        end_x = end_players.get("x")
                        end_y = end_players.get("y")
                        # remember first event data
                        if j == 0:
                            first_start = start
                        # OUT of side line
                        if y_OUT:
                            if (start_y < 0.1 or start_y > 67.9):
                                if 0.2 <= end_y <= 67.8:
                                    predictions.append({
                                        "gameTime": format_game_time(start['match_time']),
                                        "label": "THROW IN",
                                        "position": str(start['match_time']),
                                        "team": "",
                                        "confidence": "0.5"
                                    })
                                    y_OUT = False
                                    x_OUT = False
                                    skip_count = j + 1  # Skip processed events in outer loop
                                    break
                            # last event (This means it can't find event)
                            if j == 4:
                                predictions.append({
                                    "gameTime": format_game_time(first_start['match_time']),
                                    "label": "THROW IN",
                                    "position": str(start['match_time']),
                                    "team": "",
                                    "confidence": "0.5"
                                })
                                y_OUT = False
                                x_OUT = False
                        # OUT of goal line
                        elif x_OUT:
                            # GK
                            if (5.0 <= start_x <= 6.0 or 99.0 <= start_x <= 100.0) and (24 <= start_y <= 42):
                                if (5.0 <= end_x < 100.0):
                                    predictions.append({
                                        "gameTime": format_game_time(start['match_time']),
                                        "label": "GOAL KICK",
                                        "position": str(start['match_time']),
                                        "team": "",
                                        "confidence": "0.5"
                                    })
                                    y_OUT = False
                                    x_OUT = False
                                    skip_count = j + 1  # Skip processed events in outer loop
                                    break
                            # CK
                            if (start_x <= 1.5 and start_y <= 1.5) or (start_x >= 103.5 and start_y <= 1.5) or (start_x <= 1.5 and start_y >= 66.5) or (start_x >= 103.5 and start_y >= 66.5):
                                if (1.5 < end_x < 103.5) and (1.5 < end_y < 66.5):
                                    predictions.append({
                                        "gameTime": format_game_time(start['match_time']),
                                        "label": "CROSS",
                                        "position": str(start['match_time']),
                                        "team": "",
                                        "confidence": "0.5"
                                    })
                                    y_OUT = False
                                    x_OUT = False
                                    skip_count = j + 1  # Skip processed events in outer loop
                                    break
                            # GOAL
                            if (-2.0 <= end_x <= 0.0 or 105.0 <= end_x <-107.0) and (30.34 < end_y < 37.66):
                                predictions.append({
                                    "gameTime": format_game_time(start['match_time']),
                                    "label": "GOAL",
                                    "position": str(start['match_time']),
                                    "team": "",
                                    "confidence": "0.5"
                                })
                                y_OUT = False
                                x_OUT = False
                                skip_count = j + 1  # Skip processed events in outer loop
                                break
                            # last event (This means it can't find event)
                            if j == 4:
                                predictions.append({
                                    "gameTime": format_game_time(first_start['match_time']),
                                    "label": "PASS",
                                    "position": str(first_start['match_time']),
                                    "team": "",
                                    "confidence": "0.5"
                                })
                                y_OUT = False
                                x_OUT = False
                        
        elif old_label == "DRIVE":
            # Case 3: DRIVE
            predictions.append({
                "gameTime": format_game_time(start['match_time']),
                "label": "DRIVE",
                "position": str(start['match_time']),
                "team": "",
                "confidence": "0.5"
            })

    # Sort outputs by frame for chronological order
    predictions.sort(key=lambda x: int(float(x['position'])))

    # 結果をJSON形式で保存
    recognition_results = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "predictions": predictions
    }

    with open(output_json_path, 'w') as f:
        json.dump(recognition_results, f, indent=4)
    
    print(f"Data saved to {output_json_path}")

def format_game_time(time):
    """
    時間を '1 - MM:SS' の形式でフォーマットするヘルパー関数。
    """
    if time <= 2700000:
        seconds = time / 1000
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"1 - {minutes}:{seconds:02}"
    else:
        seconds = time / 1000
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"2 - {minutes}:{seconds:02}"

def ball_dis(tracking_df, output_ball_dis_path, fps=25):
    """
    Calculate the distance of all players from the ball at each match time and output in CSV.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.
        fps (int): Frames per second of the video.

    Returns:
        pd.DataFrame: DataFrame with distances of all players to the ball for each match time.
    """
    # 攻める向きも初期位置から取得し返す
    first_players_position = tracking_df[tracking_df['match_time'] == 0.0][['teamId', 'x', 'y']].reset_index(drop=True)
    if first_players_position.loc[0, 'teamId'] != None:
        first_player_teamId = first_players_position.loc[0, 'teamId']
    else:
        first_player_teamId = first_players_position.loc[1, 'teamId']
    one_side_team_players_position = first_players_position[first_players_position['teamId'] == first_player_teamId]
    centroid_x = one_side_team_players_position['x'].mean()
    if centroid_x < 52.5:
        plus_team_id = first_player_teamId

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_ball_dis_path):
        print(f"Loading cached data from {output_ball_dis_path}")
        all_distances_df = pd.read_csv(output_ball_dis_path)
        return all_distances_df.reset_index(drop=True), plus_team_id
    
    # ボールと選手データを分離
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['match_time', 'x', 'y']].reset_index(drop=True)
    player_data = tracking_df[tracking_df['id'] != 'ball'][['match_time', 'id', 'teamId', 'x', 'y']]

    # 全選手の距離データを格納するリスト
    distances_list = []

    print('start all_players_distance_to_ball.csv')

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
                'distance_to_ball': (np.sqrt((player_row['x'] - ball_x)**2 + (player_row['y'] - ball_y)**2)).round(2)
            })
        
        # 進捗を出力
        if idx % 100000 == 0:
            print(f"Processed match_time {match_time}")

    # DataFrameに変換
    all_distances_df = pd.DataFrame(distances_list)

    # CSVに保存
    all_distances_df.to_csv(output_ball_dis_path, index=False)
    print(f"Data saved to {output_ball_dis_path}")

    return all_distances_df.reset_index(drop=True), plus_team_id

def ball_player(tracking_df, all_distances_df, output_ball_player_path, ball_possess_threshold = 1.00):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_ball_player_path):
        print(f"Loading cached data from {output_ball_player_path}")
        outputs_ball_player = pd.read_json(output_ball_player_path)
        return outputs_ball_player.reset_index(drop=True)
    
    # ボールのデータのみを抽出
    ball_data = tracking_df[tracking_df['id'] == 'ball'][['frame', 'match_time', 'x', 'y']].reset_index(drop=True)
    outputs_ball_player = []
    
    for i, row in ball_data.iterrows():
        match_time = row['match_time']
        ball_x = row['x']
        ball_y = row['y']
        distances_data = all_distances_df[all_distances_df['match_time'] == match_time].reset_index(drop=True)

        # 距離が閾値以下の選手をフィルタリング
        close_players = distances_data[distances_data['distance_to_ball'] <= ball_possess_threshold]

        # OUTの記録
        if (ball_x < 0.0 or ball_x > 105.0 or ball_y < -0.1 or ball_y > 68.0): # and (last_ball_x >= 0.0 and last_ball_x <= 105.0 and last_ball_y >= 0.0 and last_ball_y <= 68.0): # and (match_time - last_match_time > 1000)
            if close_players.empty:
                players_info = [{"player_id": "OUT", "team_id": "OUT", "distance_to_ball": "OUT", "x": ball_x, "y": ball_y}]
            else:
                players_info = close_players[['player_id', 'team_id', 'distance_to_ball']].to_dict(orient="records")
                for player in players_info:
                    player['x'] = ball_x
                    player['y'] = ball_y
                players_info.append({"player_id": "OUT", "team_id": "OUT", "distance_to_ball": "OUT", "x": ball_x, "y": ball_y})

            outputs_ball_player.append({
                "match_time": match_time,
                "players": players_info
            })
            # after_OUT = True
            # last_out_match_time = match_time
        # 距離が閾値以下の選手の記録
        else:
            if close_players.empty:
                # 該当する選手がいない場合
                outputs_ball_player.append({"match_time": match_time})
            else:
                # 該当する選手がいる場合
                players_info = close_players[['player_id', 'team_id', 'distance_to_ball']].to_dict(orient="records")
                for player in players_info:
                    player['x'] = ball_x
                    player['y'] = ball_y
                outputs_ball_player.append({
                    "match_time": match_time,
                    "players": players_info
                })

    with open(output_ball_player_path, 'w') as f:
        json.dump(outputs_ball_player, f, indent=4)
    print(f"Data saved to {output_ball_player_path}")

    return pd.read_json(output_ball_player_path).reset_index(drop=True)

def group_player_data(ball_player_df, output_possession_group_path):
    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_possession_group_path):
        print(f"Loading cached data from {output_possession_group_path}")
        with open(output_possession_group_path) as f:
            possession_group_list = json.load(f)
        return possession_group_list

    possession_group_list = []
    temp_group = []

    for _, row in ball_player_df.iterrows():
        if isinstance(row.get("players"), list) or isinstance(row.get("players"), dict):
            temp_group.append(row.to_dict())
        else:
            if temp_group:
                possession_group_list.append(temp_group)
                temp_group = []

    if temp_group:  # 最後に残ったデータを追加
        possession_group_list.append(temp_group)

    with open(output_possession_group_path, "w") as f:
        json.dump(possession_group_list, f, indent=4)

    print(f"Data saved to {output_possession_group_path}")
    return possession_group_list

def player_to_player(possession_group_list, output_player_to_player_path, frame_rate, min_possess_duration):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_player_to_player_path):
        print(f"Loading cached data from {output_player_to_player_path}")
        outputs_player_to_player = pd.read_json(output_player_to_player_path)
        return outputs_player_to_player.reset_index(drop=True)
    
    outputs_player_to_player = []

    for i in range(len(possession_group_list)):
        # each group
        group_list = possession_group_list[i]

        # OUT グループか否か
        OUT_bool = False
        OUT_start = None

        for j in range(len(group_list)):
            # each frame
            frame_df = group_list[j]
            current_frame_players = frame_df["players"]

            # In progress to judge OUT or POSSESSION state
            # "players"が1つのみ
            if len(current_frame_players) == 1:
                # 1 player
                player = current_frame_players[0]
                # OUT
                if player['player_id'] == 'OUT':
                    if (player['x'] < -1.0) or (player['x'] > 106.0) or (player['y'] < -1.5) or (player['y'] > 69.0):
                        OUT_bool = True
                        OUT_start = frame_df
            # "players"が複数
            else:
                for k in range(len(current_frame_players)):
                    # 1 player
                    player = current_frame_players[k]
                    # OUT
                    if player['player_id'] == 'OUT':
                        if (player['x'] < -1.0) or (player['x'] > 106.0) or (player['y'] < -1.5) or (player['y'] > 69.0):
                            OUT_bool = True
                            OUT_start = frame_df
        # OUT グループ
        if OUT_bool:
            # 開始時刻と終了時刻
            OUT_end = group_list[len(group_list)-1]
            outputs_player_to_player.append({
                "state": "OUT",
                "start": OUT_start,
                "end": OUT_end
            })
            OUT_bool = False
            continue

        # POSSESSION グループ
        else:
            # 前回のグループがOUT
            POSSESSION_start = group_list[0]
            POSSESSION_end = group_list[len(group_list)-1]
            outputs_player_to_player.append({
                "state": "POSSESSION",
                "start": POSSESSION_start,
                "end": POSSESSION_end
            })
            continue
        
        # last_OUT_bool = OUT_bool

    # DataFrame 形式の場合
    with open(output_player_to_player_path, 'w') as f:
        json.dump(outputs_player_to_player, f, indent=4)

    print(f"Data saved to {output_player_to_player_path}")

    return pd.read_json(output_player_to_player_path).reset_index(drop=True)

def PASS_DRIVE_OUT(player_to_player_df, output_PASS_DRIVE_path):

    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(output_PASS_DRIVE_path):
        print(f"Loading cached data from {output_PASS_DRIVE_path}")
        outputs_PASS_DRIVE = pd.read_json(output_PASS_DRIVE_path)
        return outputs_PASS_DRIVE.reset_index(drop=True)
    
    outputs_PASS_DRIVE = []
    # 閾値 (方向ベクトルの角度差) を設定: 単位は度 (例: 30度以下はラベル付けしない)
    ANGLE_THRESHOLD = 20
    after_OUT = False
    DRIVE_player_id = None
    last_pass_vector = None

    # Event を記録
    for i in range(len(player_to_player_df) - 1):

        # 前の情報を取得
        if i != 0:
            previous_group_df = player_to_player_df.loc[i - 1]
            # previous_state = previous_group_df['state']
            previous_start = previous_group_df['start']
            previous_end = previous_group_df['end']
        else:
            previous_start = {
                            "match_time": 0,
                            "players": [
                                    {
                                        "player_id": 0,
                                        "team_id": 0,
                                        "x": 52.5,
                                        "y": 34,
                                        "distance_to_ball": 0
                                    }
                                ]
                            }
            previous_end = {
                            "match_time": 0,
                            "players": [
                                    {
                                        "player_id": 0,
                                        "team_id": 0,
                                        "x": 52.5,
                                        "y": 34,
                                        "distance_to_ball": 0
                                    }
                                ]
                            }
        previous_end_player = previous_end['players']
        previous_end_position = np.array([previous_end_player[0]['x'], previous_end_player[0]['y']])

        # 現在の情報を取得
        group_df = player_to_player_df.loc[i]
        current_state = group_df['state']
        current_start = group_df['start']
        current_end = group_df['end']
        current_start_player = current_start['players']
        current_end_player = current_end['players']
        current_start_position = np.array([current_start_player[0]['x'], current_start_player[0]['y']])
        current_end_position = np.array([current_end_player[0]['x'], current_end_player[0]['y']])

        # OUT を記録
        if current_state == 'OUT':
            if after_OUT:
                continue
            # ラベル付け
            label = 'PASS'
            outputs_PASS_DRIVE.append({
                "label": label,
                "start": previous_end,
                "end": current_start
            })
            after_OUT = True
        
        # PASS を記録
        else:
            if after_OUT:
                # ラベル付け
                label = 'PASS'
                outputs_PASS_DRIVE.append({
                    "label": label,
                    "start": previous_end,
                    "end": current_start
                })
                after_OUT = False
                continue
            else:
                current_pass_len = np.linalg.norm(current_start_position - previous_end_position)

                # 現在の選手と次の選手が同じ
                if previous_end_player[0]['player_id'] == current_start_player[0]['player_id']:
                    # すでにDRIVE中の選手が存在．それと比較
                    if DRIVE_player_id is not None:
                        if current_end_player[0]['player_id'] == DRIVE_player_id:
                            continue
                    DRIVE_player_id = current_start_player[0]['player_id']
                    # DRIVEの検出
                    # ラベル付け
                    label = 'DRIVE'
                    outputs_PASS_DRIVE.append({
                        "label": label,
                        "start": previous_start,
                        "end": previous_start
                    })
                # 違う選手（パスの可能性高い）
                else:
                    # 長時間持っていたらDRIVE
                    if previous_end["match_time"] - previous_start["match_time"] > 500:
                        # ラベル付け
                        label = 'DRIVE'
                        outputs_PASS_DRIVE.append({
                            "label": label,
                            "start": previous_start,
                            "end": previous_start
                        })
                    # 誤検出
                    # 前回のパスが存在する場合、方向を比較する（スルーや上空対策）
                    current_pass_vector = current_start_position - previous_end_position
                    if last_pass_vector is not None:
                        # ベクトルの角度差を計算
                        # 前回の遷移がPASSの時のみ（DRIVEの時に角度を測ると除外されてしまう）
                        if last_pass_len > 2:
                            angle_difference = get_angle_difference(last_pass_vector, current_pass_vector)
                            # 角度差が閾値以下の場合はスキップ
                            if angle_difference <= ANGLE_THRESHOLD:
                                continue

                    # ラベル付け
                    label = 'PASS'
                    outputs_PASS_DRIVE.append({
                        "label": label,
                        "start": previous_end,
                        "end": current_start
                    })
                    last_pass_vector = current_pass_vector
                last_pass_len = current_pass_len

    # Sort outputs by frame for chronological order
    # outputs_PASS_DRIVE.sort(key=lambda x: int(float(x['position'])))
    with open(output_PASS_DRIVE_path, 'w') as f:
        json.dump(outputs_PASS_DRIVE, f, indent=4)  
    print(f"Data saved to {output_PASS_DRIVE_path}")

    return pd.read_json(output_PASS_DRIVE_path).reset_index(drop=True)

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
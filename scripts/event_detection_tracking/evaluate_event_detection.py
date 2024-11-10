"""
Script to detect events in a soccer match video from tracking data in a CSV file.

This script reads tracking data of the ball and players, 
to identify events based on changes in ball movement and players position. 

The tracking data CSV structure is as follows:
frame,match_time,event_period,ball_status,id,x,y,teamId
"""

import argparse
import pandas as pd
import numpy as np
import json
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
        output_json_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_in_play_detection.json'
        # ファイルを読み込み
        with open(event_path, 'r') as f:
            events = json.load(f)
        with open(output_json_path, 'r') as f:
            outputs = json.load(f)
        label_accuracy = evaluate_event_accuracy(events, outputs)
        print(label_accuracy)

def evaluate_event_accuracy(ground_truth, recognition_results, tolerance=5000):
    """
    正解データと予測結果を比較して、ラベルごとの精度を算出する関数。

    Args:
        ground_truth (dict): 正解データのイベントを含む辞書形式のデータ。
        recognition_results (dict): 予測されたイベントを含む辞書形式のデータ。
        tolerance (int): 同一ラベルのイベントが正解と見なされる時間範囲（ミリ秒）。デフォルトは5000ミリ秒。

    Returns:
        dict: 各ラベルごとの精度を含む辞書。
    """
    # ラベルごとにイベントのフレーム位置を抽出
    ground_truth_events = defaultdict(list)
    for event in ground_truth["annotations"]:
        label = event["label"]
        ground_truth_events[label].append(float(event["position"]))

    recognition_events = defaultdict(list)
    for event in recognition_results["annotations"]:
        label = event["label"]
        recognition_events[label].append(float(event["position"]))

    # ラベルごとの正解数と総数をカウント
    label_correct_counts = defaultdict(int)
    label_total_counts = defaultdict(int)

    for label, gt_frames in ground_truth_events.items():
        rec_frames = recognition_events.get(label, [])
        correct_detections = 0

        for gt_frame in gt_frames:
            # tolerance内に同じラベルの予測値があるかを確認
            if any(abs(gt_frame - rec_frame) <= tolerance for rec_frame in rec_frames):
                correct_detections += 1

        # ラベルごとの正解数と総数を更新
        label_correct_counts[label] = correct_detections
        label_total_counts[label] = len(gt_frames)

    # ラベルごとの精度を計算
    label_accuracy = {}
    for label in label_total_counts:
        total = label_total_counts[label]
        correct = label_correct_counts[label]
        label_accuracy[label] = correct / total if total > 0 else 0

    return label_accuracy

if __name__ == '__main__':
    main()
"""
Script to filter and smooth soccer match tracking data.

This script processes tracking data of the ball and players by scaling coordinates, 
applying a Kalman filter to reduce noise over a 5-frame window, 
and outputs the filtered data in the same CSV format as the input.

The tracking data CSV structure is as follows:
frame,match_time,event_period,ball_status,id,x,y,teamId
"""

import argparse
import pandas as pd
import numpy as np
import cv2

def parse_arguments():
    """
    Parse command line arguments for match ID.

    Returns:
        argparse.Namespace: Parsed command line arguments containing match_id.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_id', help="Match ID for the video and annotation files", required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]

    for match_id in match_ids:
        tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_pitch_plane_coordinates.csv'
        output_tracking_path = f'data/interim/pitch_plane_coordinates/{match_id}/{match_id}_filtered_pitch_plane_coordinates.csv'

        # Load the tracking data
        tracking_df = pd.read_csv(tracking_path)

        # Filter and save the data
        filter_coordinates(tracking_df, output_tracking_path)

def filter_coordinates(tracking_df, output_tracking_path):
    """
    Filter and smooth tracking data, and save to CSV.

    Args:
        tracking_df (pd.DataFrame): Input tracking data.
        output_tracking_path (str): Path to save the filtered data.
    """
    # Scale coordinates
    tracking_df['x'] *= 105
    tracking_df['y'] *= 68

    # Apply Kalman filter for smoothing
    smoothed_df = apply_kalman_filter(tracking_df)

    # Replace first 5 frames with original data
    smoothed_df = replace_initial_frames(tracking_df, smoothed_df, init_frames=5)

    # 並び順を入力ファイルに合わせる
    smoothed_df = smoothed_df.sort_values(by=['match_time', 'frame']).reset_index(drop=True)

    # Round 'x' and 'y' to 2 decimal places
    smoothed_df['x'] = smoothed_df['x'].round(2)
    smoothed_df['y'] = smoothed_df['y'].round(2)

    # Save the filtered data
    smoothed_df.to_csv(output_tracking_path, index=False)

def apply_kalman_filter(data, init_frames=10):
    """
    Apply a Kalman filter to smooth x and y coordinates of tracking data for each ID, 
    using the entire time series.

    Args:
        data (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """
    unique_ids = data['id'].unique()
    smoothed_data = []

    for obj_id in unique_ids:
        print(f'Processing object ID: {obj_id}')
        obj_data = data[data['id'] == obj_id].sort_values(by='match_time')
        frames = obj_data['frame'].values
        x_values = obj_data['x'].values
        y_values = obj_data['y'].values

        # Initialize Kalman filter for combined x and y
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5
        kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # Initialize the state
        kalman.statePost = np.array([x_values[0], y_values[0], 0, 0], np.float32)

        filtered_x = []
        filtered_y = []

        for i, (x, y) in enumerate(zip(x_values, y_values)):
            # 調整可能な測定ノイズの設定
            if i < init_frames:
                # 初期フレームでは徐々に信頼度を低下させる
                alpha = (init_frames - i) / init_frames  # 信頼度の減衰率 (1 → 0)
                kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * (1e-10 * alpha + 1e-3 * (1 - alpha))
            else:
                # 通常の信頼度を適用
                kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
            # Observation (measurement) update
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            kalman.correct(measurement)

            # Time update (prediction)
            prediction = kalman.predict()

            # Append the filtered results
            filtered_x.append(prediction[0, 0])
            filtered_y.append(prediction[1, 0])

        # Update the DataFrame with smoothed values
        obj_data['x'] = [round(v, 2) for v in filtered_x]  # Round to 2 decimal places
        obj_data['y'] = [round(v, 2) for v in filtered_y]  # Round to 2 decimal places
        smoothed_data.append(obj_data)

    return pd.concat(smoothed_data)

def replace_initial_frames(original_df, smoothed_df, init_frames=5):
    """
    Replace the first `init_frames` of smoothed data with the original data.

    Args:
        original_df (pd.DataFrame): Original tracking data.
        smoothed_df (pd.DataFrame): Smoothed tracking data.
        init_frames (int): Number of initial frames to replace.

    Returns:
        pd.DataFrame: Combined DataFrame with replaced initial frames.
    """
    for obj_id in original_df['id'].unique():
        original_obj_data = original_df[original_df['id'] == obj_id]
        smoothed_obj_data = smoothed_df[smoothed_df['id'] == obj_id]

        # Identify the initial frames
        initial_frames = original_obj_data.head(init_frames)
        remaining_frames = smoothed_obj_data.iloc[init_frames:]

        # Combine original initial frames with the rest of the smoothed data
        combined_data = pd.concat([initial_frames, remaining_frames])
        smoothed_df.update(combined_data)

    return smoothed_df

if __name__ == '__main__':
    main()

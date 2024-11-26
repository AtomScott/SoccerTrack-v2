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

import pandas as pd
import argparse
import os
import json
import cv2
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
    return parser.parse_args()


def display_label(frame, label):
    """
    Display a label on a video frame.

    Args:
        frame (ndarray): The video frame to modify.
        label (str): The text label to overlay on the frame.
    """
    cv2.putText(frame, label, (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 2)


def visualize_event_video(video_path: str, events_df: pd.DataFrame, output_path: str) -> None:
    """
    Process a video and overlay event labels at specified frames based on annotations.

    Args:
        video_path (str): Path to the input video file.
        events_df (dict): Dictionary containing the event annotations.
        output_path (str): Path to save the processed video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = fps * 600  # Limit to the first 2 minutes (120 seconds).

    # Set up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_num = 0
    annotation_num = 0
    last_annotation_frame = 0
    label_display_frames = {}  # Track frames where labels should be displayed
    annotations = events_df['annotations']

    while cap.isOpened() and frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame matches any annotation's frame position
        annotation = annotations[annotation_num]
        position = float(annotation['position']) / 40.0

        # annotation frame
        if frame_num == position:
            label = annotation['label']
            # Set frames where the label will be displayed (current + next 15 frames)
            display_label(frame, label)
            annotation_num += 1
            last_annotation_frame = frame_num + 15

        # not annotation frame & current_frame < last_annotation_frame
        elif frame_num < last_annotation_frame:
            display_label(frame, label)

        # frame not needed to display
        if frame_num % 100 == 0:
            print(frame_num)

        out.write(frame)
        frame_num += 1

    # Release resources
    out.release()
    cap.release()
    logger.info(f"Created file: {output_path}")


if __name__ == '__main__':
    # Example usage
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    num_class = str(args.num_class)
    
    for match_id in match_ids:
        video_path = f'data/raw/{match_id}/{match_id}_panorama_1st_half.mp4'
        event_path = f'data/raw/{match_id}/{match_id}_{num_class}_class_events.json'
        output_path = f'data/interim/event_visualization/{match_id}/{match_id}_event_video.mp4'
        
        with open(event_path, 'r') as f:
            events_df = json.load(f)
        
        visualize_event_video(video_path, events_df, output_path)

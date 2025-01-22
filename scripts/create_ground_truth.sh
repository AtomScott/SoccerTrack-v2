#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Run the ground truth creation
uv run python -m src.main \
    command=create_ground_truth_mot \
    match.id=$MATCH_ID \
    create_ground_truth_mot.match_id=$MATCH_ID \
    create_ground_truth_mot.detections_path="data/interim/${MATCH_ID}/${MATCH_ID}_panorama_test_detections.csv" \
    create_ground_truth_mot.coordinates_path="data/interim/${MATCH_ID}/${MATCH_ID}_pitch_plane_coordinates.csv" \
    create_ground_truth_mot.homography_path="data/interim/${MATCH_ID}/${MATCH_ID}_homography.npy" \
    create_ground_truth_mot.output_path="data/interim/${MATCH_ID}/${MATCH_ID}_ground_truth_mot.csv"

# Run visualization if the ground truth was created successfully
if [ $? -eq 0 ]; then
    echo "Ground truth created successfully. Running visualization..."
    uv run python -m src.main \
        command=plot_bboxes_on_video \
        match.id=$MATCH_ID \
        plot_bboxes_on_video.match_id=$MATCH_ID \
        plot_bboxes_on_video.video_path="data/interim/${MATCH_ID}/${MATCH_ID}_panorama_test.mp4" \
        plot_bboxes_on_video.detections_path="data/interim/${MATCH_ID}/${MATCH_ID}_ground_truth_mot.csv" \
        plot_bboxes_on_video.output_path="data/interim/${MATCH_ID}/${MATCH_ID}_plot_bboxes_on_video-ground_truth_mot.mp4"
else
    echo "Failed to create ground truth MOT file"
    exit 1
fi 
#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id> [--no-ids]"
    echo "Example: $0 117093"
    echo "Example with no IDs: $0 117093 --no-ids"
    exit 1
fi

MATCH_ID=$1
SHOW_IDS=true

# Check for --no-ids flag
if [ "$2" = "--no-ids" ]; then
    SHOW_IDS=false
fi

# First, analyze bounding box dimensions and create regression models
echo "Analyzing bounding box dimensions..."
uv run python src/data_association/analyze_bbox_dimensions.py \
    --detections_path "data/interim/${MATCH_ID}/${MATCH_ID}_panorama_test_detections.csv" \
    --output_dir "data/interim/${MATCH_ID}" \
    --match_id "${MATCH_ID}" \
    --conf_threshold 0.3

# Then create ground truth MOT file using the regression models
if [ $? -eq 0 ]; then
    echo "Creating ground truth MOT file with dynamic bounding boxes..."
    uv run python -m src.main \
        command=create_ground_truth_mot_from_coordinates \
        match.id=$MATCH_ID \
        create_ground_truth_mot_from_coordinates.event_period="FIRST_HALF" \
        create_ground_truth_mot_from_coordinates.match_id=$MATCH_ID \
        create_ground_truth_mot_from_coordinates.coordinates_path="data/interim/pitch_plane_coordinates/${MATCH_ID}/${MATCH_ID}_pitch_plane_coordinates.csv" \
        create_ground_truth_mot_from_coordinates.homography_path="data/interim/homography/${MATCH_ID}/${MATCH_ID}_homography.npy" \
        create_ground_truth_mot_from_coordinates.bbox_models_path="data/interim/${MATCH_ID}/${MATCH_ID}_bbox_models.joblib" \
        create_ground_truth_mot_from_coordinates.output_path="data/interim/${MATCH_ID}/${MATCH_ID}_ground_truth_mot_dynamic_bboxes.csv"

    # Run visualization if the ground truth was created successfully
    if [ $? -eq 0 ]; then
        echo "Ground truth created successfully. Running visualization..."
        uv run python -m src.main \
            command=plot_bboxes_on_video \
            match.id=$MATCH_ID \
            plot_bboxes_on_video.match_id=$MATCH_ID \
            plot_bboxes_on_video.video_path="data/interim/${MATCH_ID}/${MATCH_ID}_panorama_test.mp4" \
            plot_bboxes_on_video.detections_path="data/interim/${MATCH_ID}/${MATCH_ID}_ground_truth_mot_dynamic_bboxes.csv" \
            plot_bboxes_on_video.output_path="data/interim/${MATCH_ID}/${MATCH_ID}_plot_bboxes_on_video-ground_truth_mot_dynamic_bboxes.mp4" \
            plot_bboxes_on_video.show_ids=$SHOW_IDS
    else
        echo "Failed to create ground truth MOT file"
        exit 1
    fi
else
    echo "Failed to analyze bounding box dimensions"
    exit 1
fi 
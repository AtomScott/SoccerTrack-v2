#!/bin/bash

RAW_DIR="/groups/gaa50073/atom/soccertrack-v2/datasets/raw"
INTERIM_DIR="/groups/gaa50073/atom/soccertrack-v2/datasets/interim"

for folder in $RAW_DIR/*; do
    if [ -d "$folder" ]; then
        FOLDER_NAME=$(basename "$folder")
        KEYPOINTS_JSON="$folder/${FOLDER_NAME}_keypoints.json"
        INPUT_VIDEO="$folder/${FOLDER_NAME}_panorama.mp4"
        INTERIM_FOLDER="$INTERIM_DIR/$FOLDER_NAME"
        INPUT_IMAGE="$INTERIM_FOLDER/frames/00_first_frame.png"
        OUTPUT_IMAGE="$INTERIM_FOLDER/frames/01_viz_keypoints.png"
        
        # Create interim directory if it doesn't exist
        mkdir -p "$INTERIM_FOLDER/frames"
        
        # Extract the first frame from the video
        ffmpeg -i "$INPUT_VIDEO" -frames:v 1 "$INPUT_IMAGE"
        
        # Run the keypoints visualization script
        python scripts/calibration/keypoints_visualization.py --keypoints_json "$KEYPOINTS_JSON" --input_image "$INPUT_IMAGE" --output_image "$OUTPUT_IMAGE"
    fi
done
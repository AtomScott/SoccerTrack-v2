#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Ensure required directories exist
mkdir -p "data/interim/${MATCH_ID}"

# Run the keypoint calibration
echo "Generating calibration mappings for match ${MATCH_ID}..."
uv run python -m src.main \
    command=generate_calibration_mappings \
    generate_calibration_mappings.match_id="$MATCH_ID" \
    generate_calibration_mappings.keypoints_path="/data/share/SoccerTrack-v2/data/raw/${MATCH_ID}/${MATCH_ID}_keypoints.json" \
    generate_calibration_mappings.video_path="/data/share/SoccerTrack-v2/data/raw/${MATCH_ID}/${MATCH_ID}_panorama_1st_half.mp4" \
    generate_calibration_mappings.output_dir="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}"
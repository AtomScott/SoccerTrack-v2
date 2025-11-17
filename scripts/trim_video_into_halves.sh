#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

BASE_PATH="/data/share/SoccerTrack-v2"
OUTPUT_DIR="${BASE_PATH}/data/interim/${MATCH_ID}"

# Run the video trimming
uv run python -m src.main \
    command=trim_video_into_halves \
    trim_video_into_halves.match_id=$MATCH_ID \
    trim_video_into_halves.input_video_path="${BASE_PATH}/data/raw/${MATCH_ID}/${MATCH_ID}_panorama.mp4" \
    trim_video_into_halves.padding_info_path="${BASE_PATH}/data/raw/${MATCH_ID}/${MATCH_ID}_padding_info.csv" \
    trim_video_into_halves.output_dir="${OUTPUT_DIR}"

# Check if trimming was successful
if [ $? -eq 0 ]; then
    echo "Successfully trimmed video into halves"

    FIRST_HALF="${OUTPUT_DIR}/${MATCH_ID}_panorama_1st_half.mp4"
    SECOND_HALF="${OUTPUT_DIR}/${MATCH_ID}_panorama_2nd_half.mp4"

    if [ -f "$FIRST_HALF" ] && [ -f "$SECOND_HALF" ]; then
        echo "Output files verified:"
        echo "  First half:  $FIRST_HALF"
        echo "  Second half: $SECOND_HALF"
    else
        echo "Warning: One or more output files are missing"
        exit 1
    fi
else
    echo "Failed to trim video"
    exit 1
fi

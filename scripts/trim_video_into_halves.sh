#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Run the video trimming
uv run python -m src.main \
    command=trim_video_into_halves \
    trim_video_into_halves.match_id=$MATCH_ID \
    trim_video_into_halves.input_video_path="data/raw/${MATCH_ID}/${MATCH_ID}_panorama.mp4" \
    trim_video_into_halves.padding_info_path="data/raw/${MATCH_ID}/${MATCH_ID}_padding_info.csv" \
    trim_video_into_halves.output_dir="data/interim/${MATCH_ID}"

# Check if trimming was successful
if [ $? -eq 0 ]; then
    echo "Successfully trimmed video into halves"
    
    # Optionally verify the output files exist
    if [ -f "data/interim/${MATCH_ID}/${MATCH_ID}_panorama_1st_half.mp4" ] && \
       [ -f "data/interim/${MATCH_ID}/${MATCH_ID}_panorama_2nd_half.mp4" ]; then
        echo "Output files verified:"
        echo "  First half: data/interim/${MATCH_ID}/${MATCH_ID}_panorama_1st_half.mp4"
        echo "  Second half: data/interim/${MATCH_ID}/${MATCH_ID}_panorama_2nd_half.mp4"
    else
        echo "Warning: One or more output files are missing"
        exit 1
    fi
else
    echo "Failed to trim video"
    exit 1
fi 
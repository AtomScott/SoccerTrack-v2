#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id> [--first-frame-only]"
    echo "Example: $0 117093"
    echo "Example with first frame only: $0 117093 --first-frame-only"
    exit 1
fi

MATCH_ID=$1
FIRST_FRAME_ONLY=false

# Check for optional --first-frame-only flag
if [ "$2" = "--first-frame-only" ]; then
    FIRST_FRAME_ONLY=true
fi

# Ensure required directories exist
mkdir -p "/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}"

# Function to calibrate a video
calibrate_video() {
    local half=$1
    local input_video="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_panorama_${half}_half.mp4"
    local output_path="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_calibrated_panorama_${half}_half"
    
    # Add appropriate extension
    if [ "$FIRST_FRAME_ONLY" = true ]; then
        output_path="${output_path}.jpg"
    else
        output_path="${output_path}.mp4"
    fi

    echo "Calibrating ${half} half video..."
    echo "Input: $input_video"
    echo "Output: $output_path"
    echo "First frame only: $FIRST_FRAME_ONLY"

    if [ "$FIRST_FRAME_ONLY" = true ]; then
        uv run python -m src.calibration.calibrate_camera_from_mappings \
            --match_id "$MATCH_ID" \
            --first_frame_only
    else
        uv run python -m src.calibration.calibrate_camera_from_mappings \
            --match_id "$MATCH_ID" \
            --input_video "$input_video" \
            --output_path "$output_path"
    fi

    if [ $? -eq 0 ]; then
        echo "Successfully calibrated ${half} half"
    else
        echo "Failed to calibrate ${half} half"
        exit 1
    fi
}

# Check if mapping files exist
if [ ! -f "/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_mapx.npy" ] || \
   [ ! -f "/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_mapy.npy" ]; then
    echo "Error: Mapping files not found. Please run keypoints calibration first:"
    echo "./scripts/calibration/generate_calibration_mappings.sh ${MATCH_ID}"
    exit 1
fi

# Check if input files exist
if [ ! -f "/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_panorama_1st_half.mp4" ]; then
    echo "Error: First half video not found at data/interim/${MATCH_ID}/${MATCH_ID}_panorama_1st_half.mp4"
    exit 1
fi

if [ ! -f "/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_panorama_2nd_half.mp4" ]; then
    echo "Error: Second half video not found at data/interim/${MATCH_ID}/${MATCH_ID}_panorama_2nd_half.mp4"
    exit 1
fi

# Calibrate both halves
calibrate_video "1st"
calibrate_video "2nd"

echo "Calibration complete for match ${MATCH_ID}" 
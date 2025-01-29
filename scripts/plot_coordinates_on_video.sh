#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id> [--first-frame-only]"
    echo "Example: $0 117093"
    echo "Options:"
    echo "  --first-frame-only  Only process the first frame and save as image"
    exit 1
fi

MATCH_ID=$1
shift  # Remove match_id from arguments

# Default values
FIRST_FRAME_ONLY="false"

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --first-frame-only)
            FIRST_FRAME_ONLY="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up base paths
BASE_DIR="data/interim/$MATCH_ID"
mkdir -p "$BASE_DIR"

# Define arrays for variants
HALVES=("first_half" "second_half")
TYPES=("calibrated" "distorted")

# Function to get video suffix based on half
get_video_suffix() {
    local half=$1
    if [ "$half" = "first_half" ]; then
        echo "1st_half"
    else
        echo "2nd_half"
    fi
}

# Process each combination
for HALF in "${HALVES[@]}"; do
    VIDEO_SUFFIX=$(get_video_suffix "$HALF")
    
    for TYPE in "${TYPES[@]}"; do
        echo "Processing $HALF $TYPE..."
        if [ "$TYPE" = "calibrated" ]; then
            VIDEO_PATH="$BASE_DIR/${MATCH_ID}_calibrated_panorama_${VIDEO_SUFFIX}.mp4"
        else
            VIDEO_PATH="$BASE_DIR/${MATCH_ID}_panorama_${VIDEO_SUFFIX}.mp4"
        fi
        
        # Set input coordinates path
        COORDINATES_PATH="$BASE_DIR/${MATCH_ID}_image_plane_coordinates_${VIDEO_SUFFIX}_${TYPE}.csv"
        
        # Set output path based on first_frame_only flag
        if [ "$FIRST_FRAME_ONLY" = "true" ]; then
            OUTPUT_PATH="$BASE_DIR/${MATCH_ID}_plot_coordinates_${VIDEO_SUFFIX}_${TYPE}.jpg"
        else
            OUTPUT_PATH="$BASE_DIR/${MATCH_ID}_plot_coordinates_${VIDEO_SUFFIX}_${TYPE}.mp4"
        fi
        
        # Run the visualization
        uv run python -m src.main \
            command=plot_coordinates_on_video \
            plot_coordinates_on_video.match_id=$MATCH_ID \
            plot_coordinates_on_video.video_path="$VIDEO_PATH" \
            plot_coordinates_on_video.coordinates_path="$COORDINATES_PATH" \
            plot_coordinates_on_video.output_path="$OUTPUT_PATH" \
            plot_coordinates_on_video.first_frame_only=$FIRST_FRAME_ONLY
        
        # Check if visualization was successful
        if [ $? -ne 0 ]; then
            echo "Failed to create visualization for $HALF $TYPE"
            exit 1
        fi
    done
done

# Print summary of outputs
echo "Successfully created visualizations:"
if [ "$FIRST_FRAME_ONLY" = "true" ]; then
    echo "Images:"
    for HALF in "${HALVES[@]}"; do
        for TYPE in "${TYPES[@]}"; do
            echo "  $BASE_DIR/${MATCH_ID}_plot_coordinates_${HALF}_${TYPE}.jpg"
        done
    done
else
    echo "Videos:"
    for HALF in "${HALVES[@]}"; do
        for TYPE in "${TYPES[@]}"; do
            echo "  $BASE_DIR/${MATCH_ID}_plot_coordinates_${HALF}_${TYPE}.mp4"
        done
    done
fi 
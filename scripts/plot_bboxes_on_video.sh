#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id> [--first-frame-only] [--no-ids]"
    echo "Example: $0 117093"
    echo "Options:"
    echo "  --first-frame-only  Only process the first frame and save as image"
    echo "  --no-ids           Don't show track IDs on bounding boxes"
    exit 1
fi

MATCH_ID=$1
shift  # Remove match_id from arguments

# Default values
FIRST_FRAME_ONLY="false"
SHOW_IDS="true"

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --first-frame-only)
            FIRST_FRAME_ONLY="true"
            shift
            ;;
        --no-ids)
            SHOW_IDS="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
        
        # Set paths based on type and half
        if [ "$TYPE" = "calibrated" ]; then
            VIDEO_PATH="data/interim/${MATCH_ID}/${MATCH_ID}_calibrated_panorama_${VIDEO_SUFFIX}.mp4"
        else
            VIDEO_PATH="data/interim/${MATCH_ID}/${MATCH_ID}_panorama_${VIDEO_SUFFIX}.mp4"
        fi
        
        DETECTIONS_PATH="data/interim/${MATCH_ID}/${MATCH_ID}_ground_truth_mot_${VIDEO_SUFFIX}_${TYPE}.csv"
        
        # Set output path based on first_frame_only flag
        if [ "$FIRST_FRAME_ONLY" = "true" ]; then
            OUTPUT_PATH="data/interim/${MATCH_ID}/${MATCH_ID}_plot_bboxes_on_video_${VIDEO_SUFFIX}_${TYPE}.jpg"
        else
            OUTPUT_PATH="data/interim/${MATCH_ID}/${MATCH_ID}_plot_bboxes_on_video_${VIDEO_SUFFIX}_${TYPE}.mp4"
        fi
        
        # Create output directory if it doesn't exist
        mkdir -p "$(dirname "$OUTPUT_PATH")"
        
        # Run the visualization
        echo "Creating visualization for $HALF $TYPE..."
        uv run python -m src.main \
            command=plot_bboxes_on_video \
            plot_bboxes_on_video.match_id=$MATCH_ID \
            plot_bboxes_on_video.video_path="$VIDEO_PATH" \
            plot_bboxes_on_video.detections_path="$DETECTIONS_PATH" \
            plot_bboxes_on_video.output_path="$OUTPUT_PATH" \
            plot_bboxes_on_video.first_frame_only=$FIRST_FRAME_ONLY \
            plot_bboxes_on_video.show_ids=$SHOW_IDS
        
        # Check if visualization was successful
        if [ $? -ne 0 ]; then
            echo "Failed to create visualization for $HALF $TYPE"
            continue
        fi
    done
done

# Print summary of outputs
echo "Successfully created visualizations:"
for HALF in "${HALVES[@]}"; do
    VIDEO_SUFFIX=$(get_video_suffix "$HALF")
    for TYPE in "${TYPES[@]}"; do
        if [ "$FIRST_FRAME_ONLY" = "true" ]; then
            echo "  data/interim/${MATCH_ID}/${MATCH_ID}_plot_bboxes_on_video_${VIDEO_SUFFIX}_${TYPE}.jpg"
        else
            echo "  data/interim/${MATCH_ID}/${MATCH_ID}_plot_bboxes_on_video_${VIDEO_SUFFIX}_${TYPE}.mp4"
        fi
    done
done 
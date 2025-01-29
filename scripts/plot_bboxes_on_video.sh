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

# Set up paths
VIDEO_PATH="data/interim/$MATCH_ID/${MATCH_ID}_panorama.mp4"
DETECTIONS_PATH="data/interim/detections/$MATCH_ID/${MATCH_ID}_detections.csv"
OUTPUT_DIR="data/interim/$MATCH_ID"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set output path based on first_frame_only flag
if [ "$FIRST_FRAME_ONLY" = "true" ]; then
    OUTPUT_PATH="$OUTPUT_DIR/${MATCH_ID}_plot_bboxes_on_video.jpg"
else
    OUTPUT_PATH="$OUTPUT_DIR/${MATCH_ID}_plot_bboxes_on_video.mp4"
fi

# Run the visualization
echo "Processing video with bounding boxes..."
uv run python -m src.main \
    command=plot_bboxes_on_video \
    plot_bboxes_on_video.match_id=$MATCH_ID \
    plot_bboxes_on_video.video_path="$VIDEO_PATH" \
    plot_bboxes_on_video.detections_path="$DETECTIONS_PATH" \
    plot_bboxes_on_video.output_path="$OUTPUT_PATH" \
    plot_bboxes_on_video.first_frame_only=$FIRST_FRAME_ONLY \
    plot_bboxes_on_video.show_ids=$SHOW_IDS

# Check if visualization was successful
if [ $? -eq 0 ]; then
    echo "Successfully created visualization:"
    echo "  Output: $OUTPUT_PATH"
else
    echo "Failed to create visualization"
    exit 1
fi 
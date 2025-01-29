#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Set up paths
BASE_DIR="data/interim/$MATCH_ID"
WEIGHTS_PATH="models/yolov8x.pt"
TRACKER_CONFIG_PATH="configs/tracker_config.yaml"

# Check if required files exist
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Error: YOLOv8 weights file not found at $WEIGHTS_PATH"
    exit 1
fi

if [ ! -f "$TRACKER_CONFIG_PATH" ]; then
    echo "Error: Tracker config file not found at $TRACKER_CONFIG_PATH"
    exit 1
fi

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
    EVENT_PERIOD=$(echo "$HALF" | tr '[:lower:]' '[:upper:]')
    
    for TYPE in "${TYPES[@]}"; do
        echo "Processing $HALF $TYPE..."
        
        # Set input video path based on type
        if [ "$TYPE" = "calibrated" ]; then
            VIDEO_PATH="$BASE_DIR/${MATCH_ID}_calibrated_panorama_${VIDEO_SUFFIX}.mp4"
        else
            VIDEO_PATH="$BASE_DIR/${MATCH_ID}_panorama_${VIDEO_SUFFIX}.mp4"
        fi
        
        # Set output path
        OUTPUT_PATH="$BASE_DIR/${MATCH_ID}_detections_${VIDEO_SUFFIX}_${TYPE}.csv"
        
        # Run the detection
        uv run python -m src.main \
            command=detect_objects \
            detect_objects.match_id=$MATCH_ID \
            detect_objects.video_path="$VIDEO_PATH" \
            detect_objects.output_dir="$BASE_DIR" \
            detect_objects.weights_path="$WEIGHTS_PATH" \
            detect_objects.tracker_config="@$TRACKER_CONFIG_PATH" \
            detect_objects.event_period=$EVENT_PERIOD
        
        # Check if detection was successful
        if [ $? -ne 0 ]; then
            echo "Failed to process $HALF $TYPE"
            exit 1
        fi
        
        # Rename the output file to include the type
        mv "$BASE_DIR/${MATCH_ID}_detections_${VIDEO_SUFFIX}.csv" "$OUTPUT_PATH"
    done
done

# Print summary of outputs
echo "Successfully generated detections:"
for HALF in "${HALVES[@]}"; do
    VIDEO_SUFFIX=$(get_video_suffix "$HALF")
    for TYPE in "${TYPES[@]}"; do
        echo "  $BASE_DIR/${MATCH_ID}_detections_${VIDEO_SUFFIX}_${TYPE}.csv"
    done
done 
#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

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

# Function to get event period based on half
get_event_period() {
    local half=$1
    if [ "$half" = "first_half" ]; then
        echo "FIRST_HALF"
    else
        echo "SECOND_HALF"
    fi
}

# Process each combination
for HALF in "${HALVES[@]}"; do
    VIDEO_SUFFIX=$(get_video_suffix "$HALF")
    EVENT_PERIOD=$(get_event_period "$HALF")
    
    for TYPE in "${TYPES[@]}"; do
        echo "Processing $HALF $TYPE..."
        
        # Set paths based on type and half
        if [ "$TYPE" = "calibrated" ]; then
            VIDEO_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_calibrated_panorama_${VIDEO_SUFFIX}.mp4"
            DETECTIONS_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_detections_${VIDEO_SUFFIX}_calibrated.csv"
        else
            VIDEO_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_panorama_${VIDEO_SUFFIX}.mp4"
            DETECTIONS_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_detections_${VIDEO_SUFFIX}_distorted.csv"
        fi
        
        COORDINATES_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_image_plane_coordinates_${VIDEO_SUFFIX}_${TYPE}.csv"
        OUTPUT_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_ground_truth_mot_${VIDEO_SUFFIX}_${TYPE}.csv"
        BBOX_MODELS_PATH="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_bbox_models_${VIDEO_SUFFIX}_${TYPE}.joblib"
        # First, analyze bounding box dimensions and create regression models
        echo "Analyzing bounding box dimensions for $HALF $TYPE..."
        uv run python src/data_association/analyze_bbox_dimensions.py \
            --detections_path "$DETECTIONS_PATH" \
            --output_path "$BBOX_MODELS_PATH" \
            --match_id "${MATCH_ID}" \
            --conf_threshold 0.3

        # For sports ball class
        uv run python src/data_association/analyze_bbox_dimensions.py \
            --detections_path "$DETECTIONS_PATH" \
            --output_path "$BASE_DIR/${MATCH_ID}_bbox_models_${VIDEO_SUFFIX}_${TYPE}_sportsball.joblib" \
            --match_id "$MATCH_ID" \
            --conf_threshold 0.3 \
            --target_class "sports ball"

        # Then create ground truth MOT file using the regression models
        if [ $? -eq 0 ]; then
            echo "Converting coordinates to bounding boxes for $HALF $TYPE..."
            uv run python -m src.main \
                command=convert_image_plane_to_bounding_box \
                match.id=$MATCH_ID \
                convert_image_plane_to_bounding_box.event_period="$EVENT_PERIOD" \
                convert_image_plane_to_bounding_box.match_id=$MATCH_ID \
                convert_image_plane_to_bounding_box.coordinates_path="$COORDINATES_PATH" \
                convert_image_plane_to_bounding_box.bbox_models_path="$BBOX_MODELS_PATH" \
                convert_image_plane_to_bounding_box.output_path="$OUTPUT_PATH"
        else
            echo "Failed to analyze bounding box dimensions for $HALF $TYPE"
            continue
        fi
    done
done
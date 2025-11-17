#!/bin/bash

# Print usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id1> [match_id2 ...]"
    echo "Example: $0 117093"
    echo "Example with multiple matches: $0 117093 117094 117095"
    exit 1
fi

# Function to check if a command was successful
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed for match ID $2"
        exit 1
    fi
}

# Function to process a single match
process_match() {
    local MATCH_ID=$1
    echo "Processing match ID: $MATCH_ID"
    
    # # Step 1: Video Preprocessing
    # echo "Step 1: Trimming video into halves..."
    # ./scripts/trim_video_into_halves.sh $MATCH_ID
    # check_success "Video trimming" $MATCH_ID
    
    # # Step 2: Coordinate Conversion
    # echo "Step 2: Converting raw XML to pitch plane coordinates..."
    # ./scripts/convert_raw_to_pitch_plane.sh $MATCH_ID
    # check_success "Raw to pitch plane conversion" $MATCH_ID
    
    # # Step 3: Camera Calibration
    # echo "Step 3: Generating calibration mappings..."
    # ./scripts/generate_calibration_mappings.sh $MATCH_ID
    # check_success "Calibration mapping generation" $MATCH_ID
    
    # # echo "Step 3b: Applying camera calibration..."
    # ./scripts/calibrate_camera.sh $MATCH_ID
    # check_success "Camera calibration" $MATCH_ID
    
    # # Step 4: Coordinate Projection
    # echo "Step 4: Converting pitch plane to image plane coordinates..."
    # ./scripts/convert_pitch_plane_to_image_plane.sh $MATCH_ID
    # check_success "Pitch plane to image plane conversion" $MATCH_ID
    
    # Step 5: Object Detection
    echo "Step 5: Running YOLOv8 detection..."
    ./scripts/generate_detections.sh $MATCH_ID
    check_success "Object detection" $MATCH_ID
    
    # Step 6: Coordinate to Bounding Box Conversion
    echo "Step 6: Converting coordinates to bounding boxes..."
    ./scripts/convert_coordinates_to_bboxes.sh $MATCH_ID
    check_success "Coordinate to bbox conversion" $MATCH_ID
    
    # Step 7: Generate Visualizations
    echo "Step 7: Generating visualizations..."
    ./scripts/plot_coordinates_on_video.sh $MATCH_ID --first-frame-only
    check_success "First frame visualization" $MATCH_ID
    
    ./scripts/plot_coordinates_on_video.sh $MATCH_ID
    check_success "Video visualization" $MATCH_ID
    
    echo "Successfully completed all steps for match ID: $MATCH_ID"
    echo "----------------------------------------"
}

# Process each match ID provided as argument
for MATCH_ID in "$@"; do
    process_match $MATCH_ID
done

# Print final summary
echo "Ground truth creation completed successfully for all matches:"
for MATCH_ID in "$@"; do
    echo "- Match ID: $MATCH_ID"
    echo "  Output directory: data/interim/$MATCH_ID/"
done

echo "
You can find the following outputs for each match:
1. Trimmed and calibrated videos (*_panorama_*.mp4)
2. Coordinate files in various formats:
   - Pitch plane coordinates (*_pitch_plane_coordinates_*.csv)
   - Image plane coordinates (*_image_plane_coordinates_*.csv)
   - Detections (*_detections_*.csv)
   - Ground truth MOT files (*_ground_truth_mot_*.csv)
3. Analysis plots and models:
   - Correlation plots (*_correlation.png)
   - Regression plots (*_regression.png)
   - Bounding box models (*_bbox_models.joblib)
4. Visualizations:
   - First frame images (*_plot_coordinates_*_half_*.jpg)
   - Full videos (*_plot_coordinates_*_half_*.mp4)
" 
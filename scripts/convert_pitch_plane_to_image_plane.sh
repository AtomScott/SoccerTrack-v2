#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Process first half - calibrated
echo "Processing first half (calibrated)..."
uv run python -m src.main \
    command=convert_pitch_plane_to_image_plane \
    convert_pitch_plane_to_image_plane.match_id=$MATCH_ID \
    convert_pitch_plane_to_image_plane.input_csv_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_pitch_plane_coordinates_1st_half.csv" \
    convert_pitch_plane_to_image_plane.homography_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_homography.npy" \
    convert_pitch_plane_to_image_plane.output_dir="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID" \
    convert_pitch_plane_to_image_plane.event_period=FIRST_HALF \
    convert_pitch_plane_to_image_plane.calibrated=true

# Check if first half calibrated conversion was successful
if [ $? -ne 0 ]; then
    echo "Failed to convert first half coordinates (calibrated)"
    exit 1
fi

# Process first half - distorted
echo "Processing first half (distorted)..."
uv run python -m src.main \
    command=convert_pitch_plane_to_image_plane \
    convert_pitch_plane_to_image_plane.match_id=$MATCH_ID \
    convert_pitch_plane_to_image_plane.input_csv_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_pitch_plane_coordinates_1st_half.csv" \
    convert_pitch_plane_to_image_plane.homography_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_homography.npy" \
    convert_pitch_plane_to_image_plane.output_dir="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID" \
    convert_pitch_plane_to_image_plane.event_period=FIRST_HALF \
    convert_pitch_plane_to_image_plane.calibrated=false \
    convert_pitch_plane_to_image_plane.camera_intrinsics_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_camera_intrinsics.npz"

# Check if first half distorted conversion was successful
if [ $? -ne 0 ]; then
    echo "Failed to convert first half coordinates (distorted)"
    exit 1
fi

# Process second half - calibrated
echo "Processing second half (calibrated)..."
uv run python -m src.main \
    command=convert_pitch_plane_to_image_plane \
    convert_pitch_plane_to_image_plane.match_id=$MATCH_ID \
    convert_pitch_plane_to_image_plane.input_csv_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_pitch_plane_coordinates_2nd_half.csv" \
    convert_pitch_plane_to_image_plane.homography_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_homography.npy" \
    convert_pitch_plane_to_image_plane.output_dir="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID" \
    convert_pitch_plane_to_image_plane.event_period=SECOND_HALF \
    convert_pitch_plane_to_image_plane.calibrated=true

# Check if second half calibrated conversion was successful
if [ $? -ne 0 ]; then
    echo "Failed to convert second half coordinates (calibrated)"
    exit 1
fi

# Process second half - distorted
echo "Processing second half (distorted)..."
uv run python -m src.main \
    command=convert_pitch_plane_to_image_plane \
    convert_pitch_plane_to_image_plane.match_id=$MATCH_ID \
    convert_pitch_plane_to_image_plane.input_csv_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_pitch_plane_coordinates_2nd_half.csv" \
    convert_pitch_plane_to_image_plane.homography_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_homography.npy" \
    convert_pitch_plane_to_image_plane.output_dir="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID" \
    convert_pitch_plane_to_image_plane.event_period=SECOND_HALF \
    convert_pitch_plane_to_image_plane.calibrated=false \
    convert_pitch_plane_to_image_plane.camera_intrinsics_path="/data/share/SoccerTrack-v2/data/interim/$MATCH_ID/${MATCH_ID}_camera_intrinsics.npz"

# Check if second half distorted conversion was successful
if [ $? -ne 0 ]; then
    echo "Failed to convert second half coordinates (distorted)"
    exit 1
fi

echo "Successfully converted coordinates for both halves (calibrated and distorted)"


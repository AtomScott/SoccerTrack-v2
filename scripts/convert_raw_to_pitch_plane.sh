#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Run the coordinate conversion
uv run python -m src.main \
    command=convert_raw_to_pitch_plane \
    convert_raw_to_pitch_plane.match_id=$MATCH_ID \
    convert_raw_to_pitch_plane.input_xml_path="data/raw/${MATCH_ID}/${MATCH_ID}_tracker_box_data.xml" \
    convert_raw_to_pitch_plane.metadata_xml_path="data/raw/${MATCH_ID}/${MATCH_ID}_tracker_box_metadata.xml" \
    convert_raw_to_pitch_plane.output_dir="data/interim/${MATCH_ID}"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "Successfully converted coordinates to pitch plane format"
    
    # Optionally verify the output file exists
    if [ -f "data/interim/${MATCH_ID}/${MATCH_ID}_pitch_plane_coordinates.csv" ]; then
        echo "Output file verified:"
        echo "  CSV: data/interim/${MATCH_ID}/${MATCH_ID}_pitch_plane_coordinates.csv"
    else
        echo "Warning: Output file is missing"
        exit 1
    fi
else
    echo "Failed to convert coordinates"
    exit 1
fi 
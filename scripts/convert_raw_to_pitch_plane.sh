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
    convert_raw_to_pitch_plane.input_xml_path="/data/share/SoccerTrack-v2/data/raw/${MATCH_ID}/${MATCH_ID}_tracker_box_data.xml" \
    convert_raw_to_pitch_plane.metadata_xml_path="/data/share/SoccerTrack-v2/data/raw/${MATCH_ID}/${MATCH_ID}_tracker_box_metadata.xml" \
    convert_raw_to_pitch_plane.output_dir="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "Successfully converted coordinates to pitch plane format"
    
    # Optionally verify the output file exists
    FIRST="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_pitch_plane_coordinates_1st_half.csv"
    SECOND="/data/share/SoccerTrack-v2/data/interim/${MATCH_ID}/${MATCH_ID}_pitch_plane_coordinates_2nd_half.csv"
    if [ -f "$FIRST" ] && [ -f "$SECOND" ]; then
        echo "Output files verified:"
        echo "  First half: $FIRST"
        echo "  Second half: $SECOND"
    else
        echo "Warning: One or more output files are missing"
        exit 1
    fi
else
    echo "Failed to convert coordinates"
    exit 1
fi 
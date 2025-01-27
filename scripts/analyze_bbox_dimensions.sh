#!/bin/bash

# Check if match_id is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <match_id>"
    echo "Example: $0 117093"
    exit 1
fi

MATCH_ID=$1

# Run the bounding box dimension analysis
uv run python src/data_association/analyze_bbox_dimensions.py \
    --detections_path "data/interim/${MATCH_ID}/${MATCH_ID}_panorama_test_detections.csv" \
    --output_dir "data/interim/${MATCH_ID}" \
    --match_id "${MATCH_ID}" \
    --conf_threshold 0.3 
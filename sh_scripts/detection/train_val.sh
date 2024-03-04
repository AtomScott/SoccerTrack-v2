#!/bin/bash

# Define the base directory for datasets and outputs
dataset_base="./datasets/roboflow-split"
output_base="./outputs/change_trainset_size"

# Loop over all directories in the roboflow-split directory
for dir in ${dataset_base}/*; do
  # Check if the directory contains a data.yaml file
  if [[ -f "${dir}/data.yaml" ]]; then
    # Extract the folder name for naming the output file
    folder_name=$(basename "${dir}")
    
    # Run the training/validation script with the found dataset
    qsub /groups/gaa50073/atom/soccertrack-v2/sh_scripts/abci/run_AG_small.sh \
      python scripts/yolov8/train_val.py \
      --input "${dir}/data.yaml" \
      --output "${output_base}/${folder_name}.json" \
      --epochs 50 \
      --imgsz 2160 \
      --batch 4 \
      --workers 8 \
      --device cuda:0
  fi
done
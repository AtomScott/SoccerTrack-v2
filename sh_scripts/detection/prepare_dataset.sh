#!/bin/bash

python scripts/roboflow/download.py

# Define the base paths and configurations
image_path="./datasets/roboflow-raw/2024-02-28/train/images"
label_path="./datasets/roboflow-raw/2024-02-28/train/labels"
config_path="./datasets/roboflow-raw/2024-02-28/data.yaml"
base_output_dir="./datasets/roboflow-split/2024-02-28"

# Arrays of train sizes and random states
trainsizes=(10 50 100)
random_states=(1 2 3 4 5)

# Loop over each train size
for train_size in "${trainsizes[@]}"; do
    # Loop over each random state
    for random_state in "${random_states[@]}"; do
        # Construct the output directory for this combination
        output_dir="${base_output_dir}-trainsize${train_size}-randomstate${random_state}"
        
        # Run the Python script with the current configuration
        python scripts/roboflow/split.py \
         --image_path "$image_path" \
         --label_path "$label_path" \
         --config_path "$config_path" \
         --train "$train_size" \
         --val 10 \
         --test 40 \
         --output_dir "$output_dir" \
         --random_state "$random_state"
    done
done
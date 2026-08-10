#!/bin/bash

# Print usage
print_usage() {
    echo "Usage: $0 <match_id> [options]"
    echo "Example: $0 117093 --skip-dataset --frame-interval 5"
    echo ""
    echo "Options:"
    echo "  --skip-dataset      Skip dataset creation"
    echo "  --skip-training     Skip model training" 
    echo "  --frame-interval N  Extract every Nth frame (default: 1)"
    echo "  --help             Show this help message"
    echo "  --device           Device to use for training (default: cuda)"
}

# Check if help is requested
if [[ "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Check if match_id is provided
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

MATCH_ID=$1
shift  # Remove match_id from arguments

# Parse flags
SKIP_DATASET=false
SKIP_TRAINING=false
FRAME_INTERVAL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Function to process one half
process_half() {
    local half=$1
    
    echo "Processing ${half} half..."
    
    # Create YOLO format dataset
    if [ "$SKIP_DATASET" = false ]; then
        echo "Creating YOLO format dataset..."
        uv run python -m src.main \
            command=create_yolo_dataset \
            create_yolo_dataset.match_id="$MATCH_ID" \
            create_yolo_dataset.half="$half" \
            create_yolo_dataset.frame_interval="$FRAME_INTERVAL"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create dataset for ${half} half"
            exit 1
        fi
    else
        echo "Skipping dataset creation..."
    fi
    
    # Train YOLO model
    if [ "$SKIP_TRAINING" = false ]; then
        echo "Training YOLO model..."
        uv run python -m src.main \
            command=train_yolo_model \
            train_yolo_model.match_id="$MATCH_ID" \
            train_yolo_model.half="$half" \
            train_yolo_model.model_type="yolov8m.pt" \
            train_yolo_model.epochs=50 \
            train_yolo_model.batch_size=16 \
            train_yolo_model.imgsz=1024 \
            train_yolo_model.name="${MATCH_ID}_${half}_half_yolo" \
            train_yolo_model.device="$DEVICE"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to train model for ${half} half"
            exit 1
        fi
    else
        echo "Skipping model training..."
    fi
    
    echo "Completed processing ${half} half"
    echo "----------------------------------------"
}

# Process both halves
process_half "1st"
process_half "2nd"

# Print completion message
echo "
Processing completed successfully for match ${MATCH_ID}!"

# Only show relevant output paths based on what was processed
if [ "$SKIP_DATASET" = false ]; then
    echo "
1. YOLO format datasets:
   - First half:  data/interim/${MATCH_ID}/ultralytics_format_1st_half_distorted/
   - Second half: data/interim/${MATCH_ID}/ultralytics_format_2nd_half_distorted/"
fi

if [ "$SKIP_TRAINING" = false ]; then
    echo "
2. Trained models:
   - Under models/soccer_player_detection/
   - Best weights will be in 'best.pt'
   - Last weights will be in 'last.pt'"
fi 
#!/bin/bash

# Print usage
print_usage() {
    echo "Usage: $0 <match_id> [options]"
    echo "Example: $0 117093 --skip-first-half --device 0"
    echo ""
    echo "Options:"
    echo "  --skip-first-half   Skip processing first half"
    echo "  --skip-second-half  Skip processing second half"
    echo "  --device N          GPU device to use (default: 0)"
    echo "  --help             Show this help message"
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
SKIP_FIRST_HALF=false
SKIP_SECOND_HALF=false
DEVICE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-first-half)
            SKIP_FIRST_HALF=true
            shift
            ;;
        --skip-second-half)
            SKIP_SECOND_HALF=true
            shift
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
    # Get absolute paths
    local input_video=$(realpath "./data/interim/${MATCH_ID}/${MATCH_ID}_panorama_${half}_half.mp4")
    local output_video=$(realpath "./data/interim/${MATCH_ID}/${MATCH_ID}_tracking_${half}_half_distorted.mp4")

    # Check if input directory exists
    if [ ! -d "$(dirname "$input_video")" ]; then
        echo "Error: Input directory does not exist: $(dirname "$input_video")"
        return 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_video")"
    
    echo "Processing ${half} half..."
    
    if [ ! -f "$input_video" ]; then
        echo "Error: Input video not found: ${input_video}"
        return 1
    fi
    
    echo "Running tracking on ${half} half..."
    cd boxmot
    poetry run python tracking/track.py \
        --tracking-method botsort \
        --reid-model clip_duke.pt \
        --yolo-model yolov8x \
        --device "$DEVICE" \
        --source "$input_video" \
        --imgsz 4096 \
        --outpath "$output_video"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to process ${half} half"
        return 1
    fi
    
    echo "Completed processing ${half} half"
    echo "Output saved to: ${output_video}"
    echo "----------------------------------------"
    cd ..
}

# Process halves based on flags
if [ "$SKIP_FIRST_HALF" = false ]; then
    process_half "1st"
fi

if [ "$SKIP_SECOND_HALF" = false ]; then
    process_half "2nd"
fi

# Print completion message
echo "
Processing completed successfully for match ${MATCH_ID}!

Output videos:
1. First half:  data/interim/${MATCH_ID}/${MATCH_ID}_tracking_1st_half.mp4
2. Second half: data/interim/${MATCH_ID}/${MATCH_ID}_tracking_2nd_half.mp4" 
# Data Processing Guide

This document describes the data processing pipeline in SoccerTrack-V2, from raw input data to final ground truth files.

## Pipeline Overview

The complete data processing pipeline consists of these steps:

1. Video Preprocessing
   - Trim videos into first and second halves
   - Extract frame timing information

2. Coordinate Processing
   - Convert raw XML tracking data to pitch coordinates
   - Project pitch coordinates to image plane
   - Generate both calibrated and distorted coordinates

3. Camera Calibration
   - Generate calibration mappings from keypoints
   - Apply calibration to videos
   - Create calibrated panorama videos

4. Object Detection
   - Run YOLOv8 detection on videos
   - Process both calibrated and distorted versions
   - Generate detection files in MOT format

5. Ground Truth Creation
   - Convert coordinates to bounding boxes
   - Generate visualization plots
   - Create final ground truth MOT files

## Running the Pipeline

### Complete Pipeline

Process an entire match with a single command:

```bash
./scripts/create_ground_truth.sh <match_id>
```

Example:
```bash
./scripts/create_ground_truth.sh 117093
```

Process multiple matches:
```bash
./scripts/create_ground_truth.sh 117093 117094 117095
```

### Individual Steps

If you need to run specific steps:

1. Video Preprocessing:
   ```bash
   ./scripts/trim_video_into_halves.sh 117093
   ```

2. Coordinate Processing:
   ```bash
   # Convert XML to pitch plane
   ./scripts/convert_raw_to_pitch_plane.sh 117093
   
   # Project to image plane
   ./scripts/convert_pitch_plane_to_image_plane.sh 117093
   ```

3. Camera Calibration:
   ```bash
   # Generate mappings
   ./scripts/calibration/generate_calibration_mappings.sh 117093
   
   # Apply calibration
   ./scripts/calibration/calibrate_camera.sh 117093
   ```

4. Object Detection:
   ```bash
   ./scripts/generate_detections.sh 117093
   ```

5. Ground Truth Creation:
   ```bash
   ./scripts/convert_coordinates_to_bboxes.sh 117093
   ```

6. Visualization:
   ```bash
   # Generate videos
   ./scripts/plot_coordinates_on_video.sh 117093
   
   # Or just first frames
   ./scripts/plot_coordinates_on_video.sh 117093 --first-frame-only
   ```

## Input Requirements

1. Raw Data Files:
   ```
   data/raw/<match_id>/
   ├── <match_id>_panorama.mp4           # Full match video
   ├── <match_id>_tracker_box_data.xml   # Raw tracking data
   ├── <match_id>_tracker_box_metadata.xml  # Metadata
   └── <match_id>_padding_info.csv       # Frame timing info
   ```

2. Configuration Files:
   ```
   configs/
   ├── default_config.yaml      # Main configuration
   ├── tracker_config.yaml      # YOLOv8 tracker settings
   └── video_trimming_config.yaml  # Video processing settings
   ```

## Output Structure

The pipeline generates files in `data/interim/<match_id>/`:

1. Videos:
   - `<match_id>_panorama_[1st/2nd]_half.mp4`
   - `<match_id>_calibrated_panorama_[1st/2nd]_half.mp4`

2. Coordinates:
   - `<match_id>_pitch_plane_coordinates_[1st/2nd]_half.csv`
   - `<match_id>_image_plane_coordinates_[1st/2nd]_half_[calibrated/distorted].csv`

3. Calibration:
   - `<match_id>_camera_intrinsics.npz`
   - `<match_id>_homography.npy`

4. Detections:
   - `<match_id>_detections_[1st/2nd]_half_[calibrated/distorted].csv`

5. Ground Truth:
   - `<match_id>_ground_truth_mot_[1st/2nd]_half_[calibrated/distorted].csv`

6. Analysis:
   - `<match_id>_[width/height]_correlation.png`
   - `<match_id>_[width/height]_regression.png`
   - `<match_id>_bbox_models.joblib`

7. Visualizations:
   - `<match_id>_plot_coordinates_[1st/2nd]_half_[calibrated/distorted].[jpg/mp4]`

## Configuration

Key configuration files and their purposes:

1. `configs/default_config.yaml`:
   - Main configuration file
   - Contains settings for all pipeline components
   - Can be overridden via command line

2. `configs/tracker_config.yaml`:
   - YOLOv8 tracking parameters
   - Detection thresholds
   - Visualization settings

3. `configs/video_trimming_config.yaml`:
   - Video processing settings
   - Frame padding configuration
   - Output video settings

## Error Handling

The pipeline includes comprehensive error checking:

1. Input Validation:
   - Checks for required files
   - Validates file formats
   - Verifies data consistency

2. Process Monitoring:
   - Tracks progress of each step
   - Provides detailed error messages
   - Allows for process resumption

3. Output Verification:
   - Validates generated files
   - Checks file integrity
   - Ensures complete output set

## Best Practices

1. Data Organization:
   - Keep raw data in `data/raw/`
   - Store intermediate files in `data/interim/`
   - Use consistent naming conventions

2. Configuration Management:
   - Document config changes
   - Use version control for configs
   - Test config modifications

3. Error Recovery:
   - Save intermediate results
   - Use error logs for debugging
   - Maintain backup copies

## Troubleshooting

Common issues and solutions:

1. Missing Files:
   - Check raw data presence
   - Verify file permissions
   - Ensure correct paths

2. Process Failures:
   - Check error messages
   - Verify configurations
   - Ensure dependencies

3. Quality Issues:
   - Validate input data
   - Check calibration quality
   - Verify detection settings 
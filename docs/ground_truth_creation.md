# Ground Truth Creation

## Overview

The ground truth creation pipeline consists of several steps:
1. Video preprocessing (trimming into halves)
2. Coordinate conversion (raw XML to pitch plane)
3. Camera calibration
4. Coordinate projection (pitch plane to image plane)
5. Object detection
6. Coordinate to bounding box conversion
7. Visualization and validation

## Video Preprocessing

Before creating ground truth data, you'll need to preprocess the match video by splitting it into first and second halves:

```bash
# First ensure the script is executable
chmod +x scripts/trim_video_into_halves.sh

# Then run with match ID
./scripts/trim_video_into_halves.sh 117093
```

### Output Files
- `data/interim/<match_id>/<match_id>_panorama_1st_half.mp4`
- `data/interim/<match_id>/<match_id>_panorama_2nd_half.mp4`

## Coordinate Conversion

Convert the raw tracking data from XML format to pitch plane coordinates:

```bash
# Convert XML tracking data to pitch plane coordinates
./scripts/convert_raw_to_pitch_plane.sh 117093
```

### Output Files
- `data/interim/<match_id>/<match_id>_pitch_plane_coordinates_1st_half.csv`
- `data/interim/<match_id>/<match_id>_pitch_plane_coordinates_2nd_half.csv`

## Camera Calibration

The calibration process involves two steps:

1. Generate calibration mappings:
```bash
./scripts/calibration/generate_calibration_mappings.sh 117093
```

2. Apply calibration to videos:
```bash
./scripts/calibration/calibrate_camera.sh 117093
```

### Output Files
- Calibration files:
  - `data/interim/<match_id>/<match_id>_camera_intrinsics.npz`
  - `data/interim/<match_id>/<match_id>_homography.npy`
- Calibrated videos:
  - `data/interim/<match_id>/<match_id>_calibrated_panorama_1st_half.mp4`
  - `data/interim/<match_id>/<match_id>_calibrated_panorama_2nd_half.mp4`

## Coordinate Projection

Project pitch plane coordinates to image plane for both calibrated and distorted views:

```bash
./scripts/convert_pitch_plane_to_image_plane.sh 117093
```

### Output Files
- First half:
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_1st_half_calibrated.csv`
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_1st_half_distorted.csv`
- Second half:
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_2nd_half_calibrated.csv`
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_2nd_half_distorted.csv`

## Object Detection

Run YOLOv8 detection on both calibrated and distorted videos:

```bash
./scripts/generate_detections.sh 117093
```

### Output Files
- First half:
  - `data/interim/<match_id>/<match_id>_detections_1st_half_calibrated.csv`
  - `data/interim/<match_id>/<match_id>_detections_1st_half_distorted.csv`
- Second half:
  - `data/interim/<match_id>/<match_id>_detections_2nd_half_calibrated.csv`
  - `data/interim/<match_id>/<match_id>_detections_2nd_half_distorted.csv`

## Coordinate to Bounding Box Conversion

Convert image plane coordinates to bounding boxes using position-based size estimation:

```bash
./scripts/convert_coordinates_to_bboxes.sh 117093
```

This step:
1. Analyzes the relationship between player position and bounding box dimensions
2. Creates regression models for width and height prediction
3. Generates ground truth MOT files with position-appropriate bounding boxes

### Output Files
- Analysis plots:
  - `data/interim/<match_id>/<match_id>_width_correlation.png`
  - `data/interim/<match_id>/<match_id>_height_correlation.png`
  - `data/interim/<match_id>/<match_id>_width_regression.png`
  - `data/interim/<match_id>/<match_id>_height_regression.png`
- Models:
  - `data/interim/<match_id>/<match_id>_bbox_models.joblib`
- Ground truth:
  - First half:
    - `data/interim/<match_id>/<match_id>_ground_truth_mot_1st_half_calibrated.csv`
    - `data/interim/<match_id>/<match_id>_ground_truth_mot_1st_half_distorted.csv`
  - Second half:
    - `data/interim/<match_id>/<match_id>_ground_truth_mot_2nd_half_calibrated.csv`
    - `data/interim/<match_id>/<match_id>_ground_truth_mot_2nd_half_distorted.csv`

## Visualization

You can visualize the coordinates on the videos to validate the results:

```bash
# Generate videos with plotted coordinates
./scripts/plot_coordinates_on_video.sh 117093

# Or generate single frame visualizations
./scripts/plot_coordinates_on_video.sh 117093 --first-frame-only
```

### Output Files
When using `--first-frame-only`:
- First half:
  - `data/interim/<match_id>/<match_id>_plot_coordinates_1st_half_calibrated.jpg`
  - `data/interim/<match_id>/<match_id>_plot_coordinates_1st_half_distorted.jpg`
- Second half:
  - `data/interim/<match_id>/<match_id>_plot_coordinates_2nd_half_calibrated.jpg`
  - `data/interim/<match_id>/<match_id>_plot_coordinates_2nd_half_distorted.jpg`

Without `--first-frame-only`:
- First half:
  - `data/interim/<match_id>/<match_id>_plot_coordinates_1st_half_calibrated.mp4`
  - `data/interim/<match_id>/<match_id>_plot_coordinates_1st_half_distorted.mp4`
- Second half:
  - `data/interim/<match_id>/<match_id>_plot_coordinates_2nd_half_calibrated.mp4`
  - `data/interim/<match_id>/<match_id>_plot_coordinates_2nd_half_distorted.mp4`

## File Structure

The pipeline expects and generates files in the following structure:
```
data/
├── raw/
│   └── <match_id>/
│       ├── <match_id>_panorama.mp4
│       ├── <match_id>_tracker_box_data.xml
│       ├── <match_id>_tracker_box_metadata.xml
│       └── <match_id>_padding_info.csv
└── interim/
    └── <match_id>/
        ├── <match_id>_panorama_1st_half.mp4
        ├── <match_id>_panorama_2nd_half.mp4
        ├── <match_id>_calibrated_panorama_1st_half.mp4
        ├── <match_id>_calibrated_panorama_2nd_half.mp4
        ├── <match_id>_camera_intrinsics.npz
        ├── <match_id>_homography.npy
        ├── <match_id>_pitch_plane_coordinates_1st_half.csv
        ├── <match_id>_pitch_plane_coordinates_2nd_half.csv
        ├── <match_id>_image_plane_coordinates_1st_half_calibrated.csv
        ├── <match_id>_image_plane_coordinates_1st_half_distorted.csv
        ├── <match_id>_image_plane_coordinates_2nd_half_calibrated.csv
        ├── <match_id>_image_plane_coordinates_2nd_half_distorted.csv
        ├── <match_id>_detections_1st_half_calibrated.csv
        ├── <match_id>_detections_1st_half_distorted.csv
        ├── <match_id>_detections_2nd_half_calibrated.csv
        ├── <match_id>_detections_2nd_half_distorted.csv
        ├── <match_id>_width_correlation.png
        ├── <match_id>_height_correlation.png
        ├── <match_id>_width_regression.png
        ├── <match_id>_height_regression.png
        ├── <match_id>_bbox_models.joblib
        ├── <match_id>_ground_truth_mot_1st_half_calibrated.csv
        ├── <match_id>_ground_truth_mot_1st_half_distorted.csv
        ├── <match_id>_ground_truth_mot_2nd_half_calibrated.csv
        └── <match_id>_ground_truth_mot_2nd_half_distorted.csv
``` 
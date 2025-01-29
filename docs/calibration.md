# Camera Calibration

This document describes the camera calibration process in SoccerTrack-V2, which is essential for accurate coordinate projection and player tracking.

## Overview

The calibration process involves two main steps:
1. Generating calibration mappings from keypoints
2. Applying the calibration to videos and coordinates

## Prerequisites

Before starting the calibration process, ensure you have:
- Raw match video split into halves (see [Video Preprocessing](ground_truth_creation.md#video-preprocessing))
- Python environment with required dependencies
- Proper file structure as described below

## File Structure

```
data/
├── raw/
│   └── <match_id>/
│       ├── <match_id>_panorama_1st_half.mp4
│       └── <match_id>_panorama_2nd_half.mp4
├── interim/
│   ├── calibrated_keypoints/
│   │   └── <match_id>/
│   │       ├── <match_id>_mapx.npy
│   │       └── <match_id>_mapy.npy
│   └── calibrated_videos/
│       └── <match_id>/
│           ├── <match_id>_panorama_1st_half.mp4
│           └── <match_id>_panorama_2nd_half.mp4
```

## Generating Calibration Mappings

The first step creates mapping files that define how to transform between different coordinate spaces:

```bash
./scripts/calibration/generate_calibration_mappings.sh <match_id>
```

This script:
1. Loads keypoint correspondences between image and pitch coordinates
2. Estimates camera parameters and distortion coefficients
3. Computes the homography matrix for coordinate projection
4. Saves the calibration files for later use

### Configuration

In `configs/default_config.yaml`:
```yaml
generate_calibration_mappings:
  min_matches: 10           # Minimum number of keypoint matches
  ransac_threshold: 3.0     # RANSAC threshold for homography estimation
  confidence: 0.99          # Confidence level for RANSAC
```

### Output Files
- `data/interim/<match_id>/<match_id>_camera_intrinsics.npz`
- `data/interim/<match_id>/<match_id>_homography.npy`

## Applying Calibration

Once the mappings are generated, you can apply them to videos:

```bash
./scripts/calibration/calibrate_camera.sh <match_id>
```

This script:
1. Loads the calibration mappings
2. Applies undistortion and perspective correction to videos
3. Generates calibrated versions of both halves

### Configuration

In `configs/default_config.yaml`:
```yaml
calibrate_camera:
  interpolation: cubic  # Interpolation method for remapping
```

### Output Files
- `data/interim/<match_id>/<match_id>_calibrated_panorama_1st_half.mp4`
- `data/interim/<match_id>/<match_id>_calibrated_panorama_2nd_half.mp4`

## Coordinate Projection

After calibration, you can project coordinates between different spaces:

```bash
./scripts/convert_pitch_plane_to_image_plane.sh <match_id>
```

This generates both calibrated and distorted image plane coordinates:
- Calibrated: Uses undistorted camera model
- Distorted: Uses original camera perspective

### Output Files
- First half:
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_1st_half_calibrated.csv`
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_1st_half_distorted.csv`
- Second half:
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_2nd_half_calibrated.csv`
  - `data/interim/<match_id>/<match_id>_image_plane_coordinates_2nd_half_distorted.csv`

## Validation

You can validate the calibration results by:
1. Visualizing projected coordinates on the video:
   ```bash
   ./scripts/plot_coordinates_on_video.sh <match_id> --first-frame-only
   ```

2. Checking the reprojection error in the calibration logs

## Troubleshooting

Common issues and solutions:

1. Poor Calibration Quality
   - Ensure enough well-distributed keypoints
   - Adjust RANSAC parameters in configuration
   - Check for outliers in keypoint correspondences

2. Distorted Output
   - Verify camera intrinsics matrix
   - Check homography matrix orientation
   - Ensure correct coordinate system conventions

3. Coordinate Projection Issues
   - Validate input coordinate ranges
   - Check for numerical precision issues
   - Verify coordinate system transformations

## Next Steps

After calibration is complete, you can proceed with:
- [Creating ground truth data](ground_truth_creation.md#dynamic-bounding-boxes)
- Running player detection and tracking
- Analyzing match data 
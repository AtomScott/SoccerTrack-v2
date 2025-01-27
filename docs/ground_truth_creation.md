# Ground Truth Creation

This guide explains how to create ground truth MOT files from pitch coordinates, including the new dynamic bounding box feature.

## Dynamic Bounding Boxes

The system can now create ground truth MOT files with dynamic bounding box sizes that adapt based on the player's position in the image. This is done through a two-step process:

1. Analyze existing detections to learn the relationship between player position and bounding box dimensions
2. Apply this learned relationship to create ground truth MOT files with position-appropriate bounding boxes

### Running the Pipeline

The entire process can be run with a single command:

```bash
./scripts/create_ground_truth_fixed_bboxes.sh <match_id>
```

For example:
```bash
./scripts/create_ground_truth_fixed_bboxes.sh 117093
```

This will:
1. Analyze detection data to learn bounding box size patterns
2. Create regression models for width and height
3. Generate a ground truth MOT file with dynamic bounding boxes
4. Create a visualization video

### Visualization Options

You can choose whether to display player IDs in the visualization:

```bash
# With player IDs (default)
./scripts/create_ground_truth_fixed_bboxes.sh 117093

# Without player IDs (uses unique colors per player)
./scripts/create_ground_truth_fixed_bboxes.sh 117093 --no-ids
```

### Output Files

The script generates several files in `data/interim/<match_id>/`:

1. Analysis plots:
   - `<match_id>_width_correlation.png`: X position vs bbox width correlation
   - `<match_id>_height_correlation.png`: Y position vs bbox height correlation
   - `<match_id>_width_regression.png`: Width prediction model fit
   - `<match_id>_height_regression.png`: Height prediction model fit

2. Model and data files:
   - `<match_id>_bbox_models.joblib`: Saved regression models for future use

3. Results:
   - `<match_id>_ground_truth_mot_dynamic_bboxes.csv`: MOT format ground truth with dynamic bboxes
   - `<match_id>_plot_bboxes_on_video-ground_truth_mot_dynamic_bboxes.mp4`: Visualization video

### Understanding the Models

The system uses:
- 2nd degree polynomial regression for width (accounts for perspective effects)
- Linear regression for height
- Data range clamping to prevent unrealistic predictions

The models automatically learn from your detection data, so they adapt to your specific camera setup and perspective. 
# Command Line Interface

SoccerTrack-V2 provides a command-line interface for all major operations. Commands are structured using the pattern:

```bash
python -m src.main command=<command-name> [options]
```

## Available Commands

### Data Processing

1. **Process Raw Data**
   ```bash
   python -m src.main command=process-raw-data match_id=<match_id>
   ```
   Process raw match data into standardized formats.

2. **Transform Coordinates**
   ```bash
   python -m src.main command=transform-coordinates match_id=<match_id>
   ```
   Convert coordinates between different coordinate spaces.

3. **Extract Features**
   ```bash
   python -m src.main command=extract-features match_id=<match_id>
   ```
   Extract features from processed data.

### Ground Truth Creation

1. **Create Ground Truth MOT**
   ```bash
   python -m src.main command=create-ground-truth-mot-from-coordinates \
     coordinates_path=data/interim/pitch_plane_coordinates/<match_id>/<match_id>_pitch_plane_coordinates.csv \
     homography_path=data/interim/homography/<match_id>/<match_id>_homography.npy \
     bbox_models_path=data/interim/<match_id>/<match_id>_bbox_models.joblib \
     output_path=data/interim/<match_id>/<match_id>_ground_truth_mot_dynamic_bboxes.csv
   ```
   Create MOT format ground truth files with dynamic bounding boxes.

### Visualization

1. **Plot Bounding Boxes**
   ```bash
   python -m src.main command=plot-bboxes-on-video \
     video_path=data/raw/<match_id>/<match_id>_panorama.mp4 \
     detections_path=data/interim/<match_id>/<match_id>_ground_truth_mot_dynamic_bboxes.csv \
     output_path=data/interim/<match_id>/<match_id>_plot_bboxes_on_video.mp4 \
     [show_ids=false]
   ```
   Visualize bounding boxes on video frames.

## Shell Scripts

The project includes shell scripts for common operations:

1. **Create Ground Truth with Fixed Bounding Boxes**
   ```bash
   ./scripts/create_ground_truth_fixed_bboxes.sh <match_id> [--no-ids]
   ```
   Create and visualize ground truth MOT files.

## Configuration Options

All commands support configuration overrides:

```bash
python -m src.main command=<command-name> [config_path=configs/custom_config.yaml] [param=value]
```

See [Configuration Guide](configuration.md) for details on available options.

## Common Parameters

- `match_id`: Match identifier (e.g., "117093")
- `config_path`: Path to configuration file
- `output_path`: Path for output files
- `show_ids`: Whether to show track IDs in visualizations (default: true)

## Error Handling

Commands will:
1. Validate input parameters
2. Check file existence
3. Create output directories if needed
4. Log progress and errors

Example error handling:
```bash
# Missing required parameter
python -m src.main command=process-raw-data
# Error: match_id is required

# Invalid match_id
python -m src.main command=process-raw-data match_id=invalid
# Error: match_id 'invalid' not found in data/raw/
``` 
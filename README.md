# SoccerTrack-V2

A comprehensive toolkit for soccer player tracking and analysis.

## Features

- Video preprocessing and calibration
- Player and ball tracking
- Coordinate conversion and projection
- Ground truth data generation
- Visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SoccerTrack-v2.git
cd SoccerTrack-v2

# Install dependencies using uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Ground Truth Creation Pipeline

The toolkit provides a complete pipeline for creating ground truth data from soccer match videos. The pipeline includes:

1. Video preprocessing (trimming into halves)
2. Coordinate conversion (raw XML to pitch plane)
3. Camera calibration
4. Coordinate projection (pitch plane to image plane)
5. Object detection using YOLOv8
6. Coordinate to bounding box conversion
7. Visualization generation

### Quick Start

Process a single match:
```bash
./scripts/create_ground_truth.sh 117093
```

Process multiple matches:
```bash
./scripts/create_ground_truth.sh 117093 117094 117095
```

### Individual Steps

If you need to run specific steps individually:

```bash
# 1. Trim video into halves
./scripts/trim_video_into_halves.sh 117093

# 2. Convert XML tracking data to pitch plane coordinates
./scripts/convert_raw_to_pitch_plane.sh 117093

# 3. Generate and apply camera calibration
./scripts/calibration/generate_calibration_mappings.sh 117093
./scripts/calibration/calibrate_camera.sh 117093

# 4. Project coordinates to image plane
./scripts/convert_pitch_plane_to_image_plane.sh 117093

# 5. Run object detection
./scripts/generate_detections.sh 117093

# 6. Convert coordinates to bounding boxes
./scripts/convert_coordinates_to_bboxes.sh 117093

# 7. Generate visualizations
./scripts/plot_coordinates_on_video.sh 117093  # Full video
./scripts/plot_coordinates_on_video.sh 117093 --first-frame-only  # Single frames
```

### Output Structure

The pipeline generates a comprehensive set of files in `data/interim/<match_id>/`:

1. Videos:
   - Trimmed halves: `*_panorama_[1st/2nd]_half.mp4`
   - Calibrated videos: `*_calibrated_panorama_[1st/2nd]_half.mp4`

2. Coordinates and Detections:
   - Pitch plane coordinates: `*_pitch_plane_coordinates_*.csv`
   - Image plane coordinates: `*_image_plane_coordinates_*.csv`
   - YOLOv8 detections: `*_detections_*.csv`
   - Ground truth MOT files: `*_ground_truth_mot_*.csv`

3. Calibration Files:
   - Camera intrinsics: `*_camera_intrinsics.npz`
   - Homography matrix: `*_homography.npy`

4. Analysis and Models:
   - Correlation plots: `*_[width/height]_correlation.png`
   - Regression plots: `*_[width/height]_regression.png`
   - Bounding box models: `*_bbox_models.joblib`

5. Visualizations:
   - First frame images: `*_plot_coordinates_*_half_*.jpg`
   - Full videos: `*_plot_coordinates_*_half_*.mp4`

## Project Structure

```
SoccerTrack-v2/
├── configs/           # Configuration files
├── data/             # Data storage (gitignored except .gitkeep)
│   ├── raw/          # Raw input data
│   └── interim/      # Intermediate processing results
├── models/           # Model storage (gitignored except .gitkeep)
├── notebooks/        # Jupyter notebooks
├── scripts/          # Shell scripts for pipeline steps
└── src/             # Core Python modules
    ├── calibration/  # Camera calibration tools
    ├── coordinate_conversion/  # Coordinate processing
    ├── detection/    # Object detection modules
    ├── visualization/  # Visualization tools
    └── video_utils/  # Video processing utilities
```

## Requirements

- Python 3.12+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Other dependencies in `requirements.txt`

## Development

- Uses [ruff](https://github.com/charliermarsh/ruff) for linting and formatting
- Line length: 120 characters
- Comprehensive type hints and docstrings required
- Git-based version control

For detailed documentation on the ground truth creation process, see [docs/ground_truth_creation.md](docs/ground_truth_creation.md).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
  - [Contents](#contents)
  - [Data Format](#data-format)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Evaluation](#evaluation)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Ground Truth Creation](docs/ground_truth_creation.md): Creating MOT files with dynamic bounding boxes
- [Visualization Guide](docs/visualization.md): Options for visualizing tracking results
- [Data Processing](docs/data_processing.md): Data preprocessing and transformation

## Overview

**SoccerTrack-V2** is an advanced dataset designed for soccer game analysis, building upon the foundation of the original SoccerTrack project. This version introduces a larger collection of full-pitch view videos, tracking data, event data, and supports the  Game State Reconstruction (GS-HOTA) evaluation metric. SoccerTrack-V2 aims to facilitate research in player tracking, event detection, game state analysis, and performance evaluation in soccer.

## Features

- **Larger Dataset:** Includes full-pitch view videos from 10 soccer matches.
- **Enhanced Evaluation Metrics:** Supports Game State Reconstruction (GS-HOTA).
- **GPS and Tracklet Matching:** Integrated GPS position data with detected player tracklets.
- **Comprehensive Annotations:** Detailed annotations covering player positions and movements.
- **Baseline Evaluation Tools:** Scripts and tools to evaluate tracking performance using standardized metrics.

## Dataset

### Repository Layout

```
SoccerTrack-V2/
├── data/
│   ├── raw/                  # Original unprocessed data
│   │   ├── {match_id}/         # Folder for each video
│   │   │   ├── {match_id}_keypoints.json  # Keypoints JSON file
│   │   │   ├── {match_id}_panorama.mp4    # Panorama image
│   │   │   ├── {match_id}_player_nodes.csv      # Event data
│   │   │   ├── {match_id}_tracker_box_data.xml    # Tracking data
│   │   │   └── {match_id}_gps.csv         # GPS data
│   │   └── ...                # Additional matches
│   ├── interim/               # Processed and cleaned data
│   │   ├── events/            # Event data
│   │   │   ├── {match_id}_events.csv
│   │   │   └── ...
│   │   ├── tracking/          # Tracking data
│   │   │   ├── {match_id}_tracking.csv
│   │   │   └── ...
│   │   ├── calibrated_video/  # Calibrated video
│   │   │   ├── match_01.mp4
│   │   │   └── ...
│   │   └── ...                # Additional interims (calibrated_videos, detection_results, image_plane_coordinates etc.)
│   └── processed/             # Annotation files
│       ├── tracking/          # Processed tracking data
│       │   ├── match_01/      # Folder for each match
│       │   │   ├── img1/      # Images
│       │   │   ├── gt/        # Ground truth
│       │   │   ├── det/       # Detections
│       │   │   └── seqinfo.ini # Sequence info
│       │   └── ...            # Additional matches
│       └── events/            # Standardized event data
│           ├── match_01.json  # Event data in JSON format
│           └── ...            # Additional matches
├── src/
│   ├── data_preprocessing/   # Scripts for data cleaning and preprocessing
│   ├── feature_extraction/   # Scripts to extract features from data
│   ├── matching/             # Scripts for GPS and tracklet matching
│   └── evaluation/           # Evaluation metric implementations
├── notebooks/                # Jupyter notebooks for analysis and experiments
├── docs/                     # Documentation and tutorials
├── .github/                  # GitHub workflows and issue templates
├── LICENSE
├── README.md
├── requirements.txt          # Python dependencies
└── setup.py                  # Installation script
```

### Contents

- **Videos:** Full-pitch view videos from 10 soccer matches in [specify format, e.g., MP4].
- **GPS Data:** Player GPS positions synchronized with video frames, provided in [specify format, e.g., CSV].
- **Annotations:** Detailed annotations for player positions, movements, and actions in [specify format, e.g., JSON].
- **Evaluation Scripts:** Tools and scripts for evaluating tracking results using GS-HOTA and other metrics.
- **Documentation:** Instructions and guidelines for using the dataset effectively.

## Usage

### Data Processing

1. **Preprocessing Data**

   Use the preprocessing scripts to prepare the data for analysis.

   ```bash
   python scripts/preprocess_data.py --input_dir ./gps_data --output_dir ./processed_data
   ```

2. **Matching GPS Data with Tracklets**

   Implement the GPS-tracklet matching as outlined in [MLSA22 Paper](https://dtai.cs.kuleuven.be/events/MLSA22/papers/MLSA22_paper_8096.pdf).

   ```bash
   python scripts/match_gps_tracklets.py --gps ./gps_data/match_01_gps.csv --tracklets ./annotations/match_01_tracklets.json --output ./matched_data/match_01_matched.json
   ```

### Evaluation

1. **Running Evaluation Metrics**

   Evaluate your tracking results using the provided GS-HOTA metric.

   ```bash
   python scripts/evaluate_tracking.py --predictions ./predictions/match_01_predictions.json --ground_truth ./annotations/match_01_ground_truth.json --metric gs_hota
   ```

2. **Generating Reports**

   Generate a comprehensive evaluation report.

   ```bash
   python scripts/generate_report.py --results ./evaluation_results --output ./reports/match_01_report.pdf
   ```

## Evaluation Metrics

### Game State Reconstruction (GS-HOTA)

GS-HOTA is an advanced evaluation metric designed to assess the quality of game state reconstructions in multi-object tracking scenarios. It considers both spatial and temporal aspects of tracking, providing a holistic measure of performance.

- **Implementation Details:**
  - Located in `scripts/evaluate_metrics/gs_hota.py`.
  - Utilizes both precision and recall components tailored for game state analysis.

- **Usage:**

  ```bash
  python scripts/evaluate_metrics/gs_hota.py --predictions predictions.json --ground_truth ground_truth.json --output results.json
  ```

## Contributing

We welcome contributions to SoccerTrack-V2! Whether it's improving documentation, adding new features, or fixing bugs, your help is appreciated.

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Create a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate tests where applicable.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use SoccerTrack-V2 in your research, please cite it as follows:

```bibtex
@article{your2024soccertrackv2,
  title={SoccerTrack-V2: An Enhanced Dataset for Soccer Game Analysis},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX-XXX},
  publisher={Publisher}
}
```

*Replace the placeholders with your actual publication details once available.*

## Contact

For questions, suggestions, or support, please contact:

- **Your Name**
- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/yourprofile)
- **Twitter:** [@YourTwitterHandle](https://twitter.com/YourTwitterHandle)

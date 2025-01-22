# SoccerTrack-V2

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

## Project Structure

- `src/`: Core source code and modules
- `configs/`: Configuration files using OmegaConf
- `scripts/`: Utility and processing scripts
- `data/`: Data storage (gitignored except `.gitkeep`)
- `models/`: Model storage (gitignored except `.gitkeep`)
- `notebooks/`: Jupyter notebooks for analysis
- `docs/`: Project documentation

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create ground truth MOT files:
   ```bash
   ./scripts/create_ground_truth_fixed_bboxes.sh <match_id>
   ```

3. Visualize results:
   ```bash
   # With player IDs
   ./scripts/create_ground_truth_fixed_bboxes.sh <match_id>
   
   # Without player IDs (unique colors)
   ./scripts/create_ground_truth_fixed_bboxes.sh <match_id> --no-ids
   ```

## Development Guidelines

- **Python Version**: Requires Python 3.12+
- **Formatting**: Uses [ruff](https://github.com/charliermarsh/ruff) for linting and formatting
- **Type Hints & Docstrings**: Comprehensive type hinting and detailed docstrings required
- **Version Control**: Git with appropriate `.gitignore` configuration

For more details, see the documentation in the `docs/` directory.

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

## Installation

Follow the steps below to set up the SoccerTrack-V2 repository on your local machine.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YourUsername/SoccerTrack-V2.git
   cd SoccerTrack-V2
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment.

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

   *Ensure you have Python 3.7 or higher installed.*

3. **Download the Dataset**

   The dataset files are large and are hosted on [your chosen hosting service, e.g., Google Drive, AWS S3]. Use the provided script to download the data.

   ```bash
   bash scripts/download_dataset.sh
   ```

   *Modify the script with your dataset's actual download links.*

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

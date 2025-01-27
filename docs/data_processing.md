# Data Processing Guide

This guide explains the data processing workflows in SoccerTrack-V2.

## Data Pipeline Overview

The data processing pipeline consists of several stages:

1. Raw Data Processing
2. Coordinate Transformation
3. Feature Extraction
4. Ground Truth Generation

### Directory Structure

```
data/
├── raw/                  # Original unprocessed data
│   └── {match_id}/      # Match-specific raw data
├── interim/             # Intermediate processed data
│   ├── events/         # Processed event data
│   ├── tracking/       # Processed tracking data
│   └── calibrated_video/ # Calibrated video files
└── processed/          # Final processed data
    ├── tracking/       # MOT format tracking data
    └── events/         # Standardized event data
```

## Processing Steps

### 1. Raw Data Processing

Process raw match data using:
```bash
python -m src.main command=process-raw-data match_id=<match_id>
```

This step:
- Converts raw XML tracking data to CSV
- Processes GPS coordinate data
- Standardizes event data format

### 2. Coordinate Transformation

Transform coordinates between different spaces:
```bash
python -m src.main command=transform-coordinates match_id=<match_id>
```

Supported transformations:
- Image plane ↔ Pitch plane
- GPS coordinates ↔ Pitch plane

### 3. Feature Extraction

Extract features from processed data:
```bash
python -m src.main command=extract-features match_id=<match_id>
```

Features include:
- Player positions and velocities
- Team formations
- Event-related features

### 4. Ground Truth Generation

Generate ground truth MOT files:
```bash
./scripts/create_ground_truth_fixed_bboxes.sh <match_id>
```

See [Ground Truth Creation](ground_truth_creation.md) for detailed instructions.

## Configuration

All processing steps use configuration files in the `configs/` directory:
- `default_config.yaml`: Base configuration
- `data_processing_config.yaml`: Data processing specific settings

See [Configuration Guide](configuration.md) for details. 
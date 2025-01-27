# Configuration Guide

SoccerTrack-V2 uses OmegaConf for configuration management, providing a flexible and hierarchical configuration system.

## Configuration Files

The project uses several configuration files:

```
configs/
├── default_config.yaml     # Base configuration
├── data_processing_config.yaml  # Data processing settings
└── visualization_config.yaml    # Visualization settings
```

## Configuration Structure

### Base Configuration

```yaml
# configs/default_config.yaml
defaults:
  - data_processing
  - visualization

paths:
  data_dir: data
  raw_dir: ${paths.data_dir}/raw
  interim_dir: ${paths.data_dir}/interim
  processed_dir: ${paths.data_dir}/processed

processing:
  confidence_threshold: 0.3
  min_track_length: 10
  max_track_gap: 30

visualization:
  bbox_line_width: 2
  font_size: 1
  font_thickness: 2
```

### Command-Specific Configuration

Each command can have its own configuration section:

```yaml
# Example command configuration
create_ground_truth_mot_from_coordinates:
  coordinates_path: ${paths.interim_dir}/pitch_plane_coordinates/${match_id}/${match_id}_pitch_plane_coordinates.csv
  homography_path: ${paths.interim_dir}/homography/${match_id}/${match_id}_homography.npy
  bbox_models_path: ${paths.interim_dir}/${match_id}/${match_id}_bbox_models.joblib
  output_path: ${paths.interim_dir}/${match_id}/${match_id}_ground_truth_mot_dynamic_bboxes.csv
```

## Using Configurations

### Command Line Override

Override configuration values from the command line:

```bash
python -m src.main command=process-raw-data processing.confidence_threshold=0.5
```

### Configuration Inheritance

Configurations can inherit and override values:

```yaml
# configs/custom_config.yaml
defaults:
  - default_config

processing:
  confidence_threshold: 0.4  # Override default value
```

### Environment Variables

Use environment variables in configuration:

```yaml
paths:
  data_dir: ${oc.env:DATA_DIR,data}  # Use DATA_DIR env var or 'data' as default
```

## Best Practices

1. **Default Values**
   - Always provide sensible defaults
   - Document expected value ranges

2. **Path Management**
   - Use relative paths when possible
   - Reference paths using interpolation: `${paths.data_dir}/raw`

3. **Command Configuration**
   - Group related settings under command names
   - Use descriptive parameter names

4. **Type Safety**
   - Use OmegaConf's structured configs for type validation
   - Define expected types for configuration values

## Example Usage

```python
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('configs/default_config.yaml')

# Access values
data_dir = config.paths.data_dir
threshold = config.processing.confidence_threshold

# Override values
config.processing.confidence_threshold = 0.5

# Save modified configuration
OmegaConf.save(config, 'configs/modified_config.yaml')
``` 
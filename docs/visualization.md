# Visualization Guide

This guide covers the visualization options available in SoccerTrack-V2.

## Bounding Box Visualization

The system provides flexible options for visualizing bounding boxes on video frames.

### Player Identification Options

You can choose between two visualization modes:

1. **With Player IDs (Default)**
   - Bounding boxes are colored by team
   - Player IDs are displayed above each box
   - Track IDs use unique colors for easy tracking

2. **Without Player IDs**
   - Each player gets a unique color
   - No ID labels are shown
   - Colors are distributed across HSV space for maximum distinction

### Running Visualizations

Use the visualization script with your preferred options:

```bash
# Default visualization with IDs
./scripts/create_ground_truth_fixed_bboxes.sh 117093

# Visualization without IDs (unique colors)
./scripts/create_ground_truth_fixed_bboxes.sh 117093 --no-ids
```

### Output Files

Visualization outputs are saved in `data/interim/<match_id>/`:
- `<match_id>_plot_bboxes_on_video-ground_truth_mot_dynamic_bboxes.mp4`: Main visualization video
- Additional analysis plots if running with ground truth creation

### Customization

The visualization system supports:
- Team-based coloring
- Unique track ID colors
- Dynamic bounding box sizes
- Optional ID labels
- Automatic color generation for player distinction 
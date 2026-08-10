"""Plot pitch plane coordinates on a football field visualization."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from mplsoccer import Pitch


def plot_pitch_coordinates(
    coordinates_path: Path | str,
    output_path: Path | str,
    frame_number: int | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """
    Plot pitch plane coordinates on a football field visualization.
    
    Args:
        coordinates_path: Path to the CSV file containing pitch plane coordinates
        output_path: Path to save the output image
        frame_number: Specific frame to plot (if None, plots first frame)
        pitch_length: Length of the pitch in meters
        pitch_width: Width of the pitch in meters
    """
    logger.info(f"Loading coordinates from: {coordinates_path}")
    
    # Load coordinates
    coordinates_df = pd.read_csv(coordinates_path)
    logger.info(f"Loaded {len(coordinates_df)} coordinate entries")
    
    # Filter for specific frame if requested
    if frame_number is not None:
        coordinates_df = coordinates_df[coordinates_df["frame"] == frame_number]
    else:
        # Use the first available frame
        frame_number = coordinates_df["frame"].min()
        coordinates_df = coordinates_df[coordinates_df["frame"] == frame_number]
    
    logger.info(f"Plotting frame {frame_number} with {len(coordinates_df)} points")
    
    # Create pitch
    pitch = Pitch(
        pitch_type='custom',
        pitch_color='grass',
        line_color='white',
        pitch_width=pitch_width,
        pitch_length=pitch_length,
        goal_type='box',
        linewidth=2
    )
    
    fig, ax = pitch.draw(figsize=(16, 10))
    
    # Define colors for teams and ball
    team_colors = {
        9701: '#0080FF',  # Blue team
        9834: '#FF8000',  # Orange team
        'ball': '#FFFF00'  # Yellow for ball
    }
    
    # Plot players by team
    for team_id in coordinates_df['teamId'].unique():
        if pd.notna(team_id):  # Skip NaN values
            team_data = coordinates_df[coordinates_df['teamId'] == team_id]
            # Convert normalized coordinates to pitch coordinates
            x_coords = team_data['x'].values * pitch_length
            y_coords = team_data['y'].values * pitch_width
            
            color = team_colors.get(int(team_id), '#FFFFFF')  # Default white
            pitch.scatter(
                x_coords, y_coords,
                ax=ax,
                color=color,
                s=200,
                edgecolors='black',
                linewidth=1,
                label=f'Team {int(team_id)}'
            )
    
    # Plot ball separately if it exists
    ball_data = coordinates_df[coordinates_df['id'] == 'ball']
    if not ball_data.empty:
        x_ball = ball_data['x'].values[0] * pitch_length
        y_ball = ball_data['y'].values[0] * pitch_width
        pitch.scatter(
            x_ball, y_ball,
            ax=ax,
            color=team_colors['ball'],
            s=150,
            edgecolors='black',
            linewidth=2,
            marker='o',
            label='Ball'
        )
    
    # Add title and legend
    plt.title(f'Pitch Coordinates - Frame {frame_number}', fontsize=16, pad=20)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {output_path}")
    plt.close()


def main():
    """Main function to parse arguments and execute plotting."""
    parser = argparse.ArgumentParser(description="Plot pitch plane coordinates on a football field")
    parser.add_argument(
        "--coordinates_path",
        type=str,
        required=True,
        help="Path to the pitch plane coordinates CSV file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/pitch_coordinates_plot.png",
        help="Path to save the output image"
    )
    parser.add_argument(
        "--frame_number",
        type=int,
        default=None,
        help="Specific frame number to plot (default: first frame)"
    )
    parser.add_argument(
        "--pitch_length",
        type=float,
        default=105.0,
        help="Pitch length in meters"
    )
    parser.add_argument(
        "--pitch_width",
        type=float,
        default=68.0,
        help="Pitch width in meters"
    )
    
    args = parser.parse_args()
    
    plot_pitch_coordinates(
        coordinates_path=args.coordinates_path,
        output_path=args.output_path,
        frame_number=args.frame_number,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
    )


if __name__ == "__main__":
    main() 
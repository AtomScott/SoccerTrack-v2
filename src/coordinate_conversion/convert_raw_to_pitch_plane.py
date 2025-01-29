"""Functions for converting raw XML tracking data to pitch plane coordinates.

Usage:
    There are two ways to run the coordinate conversion:

    1. Using the Python module directly:
    ```bash
    python -m src.main command=convert_raw_to_pitch_plane \\
        convert_raw_to_pitch_plane.match_id=117093 \\
        convert_raw_to_pitch_plane.input_xml_path="data/raw/117093/117093_tracker_box_data.xml" \\
        convert_raw_to_pitch_plane.metadata_xml_path="data/raw/117093/117093_tracker_box_metadata.xml" \\
        convert_raw_to_pitch_plane.output_dir="data/interim/pitch_plane_coordinates/117093"
    ```

    2. Using the convenience shell script (recommended):
    ```bash
    # First ensure the script is executable
    chmod +x scripts/convert_raw_to_pitch_plane.sh
    
    # Then run with match ID
    ./scripts/convert_raw_to_pitch_plane.sh 117093
    ```
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import pandas as pd

from loguru import logger

VALID_EVENT_PERIODS = {"FIRST_HALF", "SECOND_HALF"}
EVENT_PERIOD_TO_FILENAME = {"FIRST_HALF": "1st_half", "SECOND_HALF": "2nd_half"}


def parse_xml(xml_path: Path | str, metadata_path: Path | str) -> List[Dict]:
    """
    Parse the XML file and extract tracking data.

    Args:
        xml_path: Path to the XML file.
        metadata_path: Path to the metadata XML file.

    Returns:
        List of dictionaries containing tracking information for each player in each frame.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracking_data = []

    # Load metadata and create a mapping of playerId to teamId
    team_mapping = {}
    metadata_tree = ET.parse(metadata_path)
    metadata_root = metadata_tree.getroot()

    for player in metadata_root.findall(".//player"):
        player_id = player.get("id")
        team_id = player.get("teamId")
        team_mapping[player_id] = team_id

    for frame in root.findall("frame"):
        frame_number = int(frame.get("frameNumber"))
        match_time = float(frame.get("matchTime"))
        event_period = frame.get("eventPeriod")

        # Skip frames with invalid event periods
        if event_period not in VALID_EVENT_PERIODS:
            logger.warning(f"Invalid event period {event_period} in frame {frame_number}, skipping")
            continue

        ball_status = frame.get("ballStatus")

        for player in frame.findall("player"):
            player_id = player.get("playerId")
            loc = player.get("loc")
            # Convert loc string to float coordinates
            try:
                x, y = map(float, loc.strip("[]").split(","))
                tracking_data.append(
                    {
                        "frame": frame_number,
                        "match_time": match_time,
                        "event_period": event_period,
                        "ball_status": ball_status,
                        "id": player_id,
                        "x": x,
                        "y": y,
                        "teamId": team_mapping.get(player_id),
                    }
                )
            except ValueError:
                logger.warning(f"Invalid location format for player {player_id} in frame {frame_number}")

        for ball in frame.findall("ball"):
            ball_id = ball.get("playerId")
            loc = ball.get("loc")
            # Convert loc string to float coordinates
            try:
                x, y = map(float, loc.strip("[]").split(","))
                tracking_data.append(
                    {
                        "frame": frame_number,
                        "match_time": match_time,
                        "event_period": event_period,
                        "ball_status": ball_status,
                        "id": ball_id,
                        "x": x,
                        "y": y,
                        "teamId": None,
                    }
                )
            except ValueError:
                logger.warning(f"Invalid location format for ball in frame {frame_number}")

    return tracking_data


def write_csv(tracking_data: List[Dict], output_csv: Path | str) -> None:
    """
    Write the tracking data to a CSV file in MOT-style format.

    Args:
        tracking_data: List of dictionaries containing tracking information.
        output_csv: Path to the output CSV file.
    """
    fieldnames = ["frame", "match_time", "event_period", "ball_status", "id", "x", "y", "teamId"]
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in tracking_data:
            writer.writerow(data)

    logger.info(f"Tracking data written to {output_csv}")


def convert_raw_to_pitch_plane(
    match_id: str,
    input_xml_path: Path | str,
    metadata_xml_path: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """
    Convert raw XML tracking data to pitch plane coordinates CSV.

    Args:
        match_id: Match ID for naming output files.
        input_xml_path: Path to the input XML file.
        metadata_xml_path: Path to the metadata XML file.
        output_dir: Directory to save the output CSV.

    Returns:
        Tuple of paths to the output CSV files (first half, second half).

    Raises:
        FileNotFoundError: If input files don't exist.
    """
    try:
        # Convert paths to Path objects
        input_xml_path = Path(input_xml_path)
        metadata_xml_path = Path(metadata_xml_path)
        output_dir = Path(output_dir)

        # Check input files exist
        if not input_xml_path.exists():
            raise FileNotFoundError(f"Input XML file not found: {input_xml_path}")
        if not metadata_xml_path.exists():
            raise FileNotFoundError(f"Metadata XML file not found: {metadata_xml_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse XML and convert to tracking data
        logger.info(f"Parsing XML file: {input_xml_path}")
        tracking_data = parse_xml(input_xml_path, metadata_xml_path)

        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(tracking_data)

        output_files = []

        # Process each half separately
        for event_period in VALID_EVENT_PERIODS:
            half_data = df[df["event_period"] == event_period].to_dict("records")
            if not half_data:
                logger.warning(f"No data found for {event_period}")
                continue

            output_csv = output_dir / f"{match_id}_pitch_plane_coordinates_{EVENT_PERIOD_TO_FILENAME[event_period]}.csv"
            write_csv(half_data, output_csv)
            output_files.append(output_csv)
            logger.info(f"Successfully created pitch plane coordinates CSV for {event_period}: {output_csv}")

        if len(output_files) != 2:
            logger.warning("Did not find data for both halves")

        return tuple(output_files)

    except FileNotFoundError as e:
        logger.error(str(e))
        raise

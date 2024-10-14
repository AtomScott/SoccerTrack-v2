"""
This script converts raw XML tracking data to MOT-style CSV files using pitch plane coordinates.

Usage:
    python coordinate_conversion/convert_raw_to_pitch_plane_mot.py --match_id <match_id>

Example:
    python coordinate_conversion/convert_raw_to_pitch_plane_mot.py --match_id 117093

Arguments:
    --match_id: The match identifier to process.
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from loguru import logger


def parse_xml(xml_path):
    """
    Parse the XML file and extract tracking data.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        list of dict: A list containing tracking information for each player in each frame.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracking_data = []

    for frame in root.findall("frame"):
        frame_number = int(frame.get("frameNumber"))
        for player in frame.findall("player"):
            player_id = player.get("playerId")
            loc = player.get("loc")
            # Convert loc string to float coordinates
            try:
                x, y = map(float, loc.strip("[]").split(","))
                tracking_data.append({"frame": frame_number, "id": player_id, "x": x, "y": y})
            except ValueError:
                logger.warning(f"Invalid location format for player {player_id} in frame {frame_number}")

    return tracking_data


def write_csv(tracking_data, output_csv):
    """
    Write the tracking data to a CSV file in MOT-style format.

    Args:
        tracking_data (list of dict): Tracking information.
        output_csv (str): Path to the output CSV file.
    """
    fieldnames = ["frame", "id", "x", "y"]
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in tracking_data:
            writer.writerow(data)

    logger.info(f"Tracking data written to {output_csv}")


def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(description="Convert raw XML tracking data to MOT-style CSV using match_id.")
    parser.add_argument("--match_id", required=True, type=str, help="The match identifier to process.")
    args = parser.parse_args()

    match_id = args.match_id
    repo_root = Path(__file__).parent.parent.parent  # Assuming script is in src/coordinate_conversion/
    input_xml = repo_root / "data" / "raw" / match_id / f"{match_id}_tracker_box_data.xml"
    output_csv = repo_root / "data" / "interim" / "pitch_plane_coordinates" / f"{match_id}_pitch_plane_coordinates.csv"

    if not input_xml.exists():
        logger.error(f"Input XML file does not exist: {input_xml}")
        return

    logger.info(f"Parsing XML file: {input_xml}")
    tracking_data = parse_xml(str(input_xml))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing tracking data to CSV: {output_csv}")
    write_csv(tracking_data, str(output_csv))


if __name__ == "__main__":
    main()

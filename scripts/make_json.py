"""
Script to convert event data from CSV format to JSON format with specific event classifications.

This script reads event data from CSV files and translates them into a standardized JSON format
for soccer event annotations. It supports multiple match IDs and different classification schemes
(e.g., 12-class, 14-class).

The output JSON follows the structure:
{
    "UrlLocal": "",
    "UrlYoutube": "",
    "annotations": [
        {
            "gameTime": "1 - mm:ss",
            "label": "event_label",
            "position": "event_time_ms",
            "team": "",
            "visibility": ""
        },
        ...
    ]
}
"""

import pandas as pd
import argparse
import os
from collections import Counter
import json
from loguru import logger


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments containing match_id and num_class.
    """
    parser = argparse.ArgumentParser(description="Convert event data from CSV to JSON format")
    parser.add_argument("--match_id", required=True, help="Comma-separated list of match IDs")
    parser.add_argument("--num_class", required=True, help="Number of event classes (e.g., '12', '14')")
    return parser.parse_args()


def make_json(raw_data_path: str, translate_event_df: pd.DataFrame, num_class: str, match_id: str) -> None:
    """
    Convert event data from CSV to JSON format for a specific match.

    Args:
        raw_data_path (str): Base path to the raw data directory.
        translate_event_df (pd.DataFrame): DataFrame containing event translation mappings.
        num_class (str): Number of event classes to use for classification.
        match_id (str): ID of the match to process.

    Returns:
        None: Writes the JSON file to disk.
    """
    # Read the event data
    event_path = os.path.join(raw_data_path, match_id, f"{match_id}_player_nodes.csv")
    event_df = pd.read_csv(event_path)
    output_json_path = os.path.join(raw_data_path, match_id, f"{match_id}_{num_class}_class_events.json")

    # Initialize JSON data structure
    json_data = {"UrlLocal": "", "UrlYoutube": "", "annotations": []}

    # Process each row in the event DataFrame
    for _, row in event_df.iterrows():
        event_type = row["event_types"]
        event_time = row["event_time"]
        event_period = row["event_period"]

        # Find matching event in translation DataFrame
        matched_event = translate_event_df[translate_event_df["Event"] == str(event_type).split(" ")[0]]

        if (not matched_event.empty) and (matched_event[num_class + "_class_event"] != "Nan").any().any():
            label = matched_event[num_class + "_class_event"].values[0]

            # Convert event time to minutes:seconds format
            total_seconds = event_time / 1000
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)

            # Format game time based on period
            if event_period == "FIRST_HALF":
                game_time = f"1 - {minutes}:{seconds:02d}"
            elif event_period == "SECOND_HALF":
                game_time = f"2 - {minutes}:{seconds:02d}"
            else:
                game_time = f"{minutes}:{seconds:02d}"

            # Create annotation entry
            annotation = {
                "gameTime": game_time,
                "label": label,
                "position": str(event_time),
                "team": "",
                "visibility": "",
            }

            json_data["annotations"].append(annotation)

    # Write JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    logger.info(f"Created JSON file: {output_json_path}")


if __name__ == "__main__":
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    num_class = str(args.num_class)
    raw_data_path = "/home/atom/SoccerTrack-v2/data/raw"
    translate_csv_path = "/home/atom/SoccerTrack-v2/data/interim/event_frequency.csv"
    translate_event_df = pd.read_csv(translate_csv_path)

    for match_id in match_ids:
        make_json(raw_data_path, translate_event_df, num_class, match_id)

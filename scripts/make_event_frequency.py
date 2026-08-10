"""
    Script to count event type combinations across matches and save their frequencies to a single CSV file.

    This script processes event data for multiple matches, calculates the frequency of each unique
    event type combination in the 'filtered_event_types' column, and outputs the results into a single CSV file.
"""

import pandas as pd
import argparse
import os
from collections import Counter
from loguru import logger

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments containing match_id.
    """
    parser = argparse.ArgumentParser(description="Count event combinations and output their frequencies.")
    parser.add_argument("--match_id", required=True, help="Comma-separated list of match IDs")
    return parser.parse_args()

def main():
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    raw_data_path = "data/raw"
    output_csv_path = "data/interim/filtered_event_frequency.csv"

    # Initialize an empty DataFrame for global event frequencies
    event_frequency_df = pd.DataFrame(columns=["Event", "Frequency"])

    for match_id in match_ids:
        # Read the event data for the match
        original_event_path = os.path.join(raw_data_path, match_id, f"{match_id}_player_nodes.csv")
        original_event_df = pd.read_csv(original_event_path)

        # Update event frequency DataFrame with the current match's data
        event_frequency_df = make_event_frequency(original_event_df, event_frequency_df)

    # Save the final event frequency DataFrame to CSV
    event_frequency_df.sort_values(by="Frequency", ascending=False, inplace=True)
    event_frequency_df.to_csv(output_csv_path, index=False)
    logger.info(f"Global CSV file created at: {output_csv_path}")

def make_event_frequency(original_event_df: pd.DataFrame, event_frequency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update event frequency DataFrame with unique combinations and their counts from the current match.

    Args:
        original_event_df (pd.DataFrame): DataFrame containing the original event data for a match.
        event_frequency_df (pd.DataFrame): DataFrame to store global event frequencies.

    Returns:
        pd.DataFrame: Updated event frequency DataFrame.
    """
    # Extract and normalize event combinations
    event_combinations = original_event_df['filtered_event_types']

    # Count frequencies in the current match
    match_frequency = event_combinations.value_counts().reset_index()
    match_frequency.columns = ["Event", "Frequency"]

    # Merge with the global event frequency DataFrame
    if event_frequency_df.empty:
        event_frequency_df = match_frequency
    else:
        event_frequency_df = pd.concat([event_frequency_df, match_frequency])
        event_frequency_df = event_frequency_df.groupby("Event", as_index=False).sum()

    return event_frequency_df

if __name__ == "__main__":
    main()

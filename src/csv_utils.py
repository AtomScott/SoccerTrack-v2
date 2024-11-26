"""Utility functions for data association and processing."""

import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger


def load_detections(detections_path: Path) -> pd.DataFrame:
    """
    Load detection results from CSV.

    Args:
        detections_path (Path): Path to the CSV file containing detections.

    Returns:
        pd.DataFrame: Detections dataframe.
    """
    logger.info(f"Loading detections from: {detections_path}")
    print(pd.read_csv(detections_path).head())
    detections = pd.read_csv(
        detections_path,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z", "class_name"],
    )
    detections["frame"] = detections["frame"].astype(int)
    return detections


def load_coordinates(coordinates_path: Path, match_id: str, event_period: Optional[str] = None) -> pd.DataFrame:
    """
    Load pitch plane coordinates from CSV.

    Args:
        coordinates_path (Path): Path to the CSV file containing coordinates.
        match_id (str): The ID of the match.
        event_period (str | None): Event period to filter coordinates (e.g., 'FIRST_HALF').

    Returns:
        pd.DataFrame: Filtered coordinates dataframe.
    """
    logger.info(f"Loading coordinates from: {coordinates_path}")
    coordinates = pd.read_csv(coordinates_path)
    coordinates["frame"] = coordinates["frame"].astype(int) - coordinates["frame"].min()

    if event_period:
        logger.info(f"Filtering coordinates for event period: {event_period}")
        coordinates = coordinates[coordinates["event_period"] == event_period]

    return coordinates

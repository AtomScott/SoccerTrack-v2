import argparse
from typing import Dict, Any, Optional
import xmltodict
from pydantic import ValidationError
from tqdm import tqdm

# from sportslabkit.dataframe import CoordinatesDataFrame
from dataclass import MatchMetadata, Frame, Player

def read_tracker_box_data(file_path: str) -> Dict[str, Any]:
    """
    Reads an XML file and converts it to a dictionary.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        Dict[str, Any]: The XML file converted to a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_dict = xmltodict.parse(file.read(), process_namespaces=True)
    
    frames = []
    for frame_dict in tqdm(xml_dict['data']['frame'], desc='Loading frames'):
        frame = Frame(**frame_dict)
        frames.append(frame)
    return frames

def read_tracker_box_metadata(file_path: str) -> Dict[str, Any]:
    """
    Reads an XML file and converts it to a dictionary.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        Dict[str, Any]: The XML file converted to a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_dict = xmltodict.parse(file.read(), process_namespaces=True)
    metadata_dict = xml_dict['metadata']
    print(metadata_dict['teams'])
    print(metadata_dict.keys())
    MatchMetadata(**metadata_dict)
    return xml_dict

def save_tracker_box_data_to_slk(tracker_box_data):
    arr = None # (L, N, 2)
    teams_ids = []
    player_ids = []
    attributes = ["x", "y"]
    auto_fix_columns=False 
    codf = CoordinatesDataFrame.from_numpy(arr, team_ids, player_ids, attributes, auto_fix_columns)

def main(data_file_path: str, metadata_file_path: str):
    """
    Main function to read XML, convert to dict, and parse using Pydantic.

    Args:
        file_path (str): The path to the XML file.
    """

    # tracker_box_data = read_tracker_box_data(data_file_path)
    tracker_box_metadata = read_tracker_box_metadata(metadata_file_path)
    return tracker_box_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and process an XML file.")
    parser.add_argument("data_file_path", type=str, help="The path to the XML file to read.")
    parser.add_argument("metadata_file_path", type=str, default=None, help="The path to the metadata XML file to read. Optional.")
    args = parser.parse_args()

    tracker_box_data = main(args.data_file_path, args.metadata_file_path)
import argparse
import os
from dotenv import load_dotenv
from roboflow import Roboflow
from pathlib import Path
from loguru import logger
from datetime import datetime


def download_dataset(api_key, project_id, location):
    """
    Downloads the dataset using the Roboflow API.

    Args:
        api_key (str): The API key for Roboflow.
        project_id (str): The project ID in Roboflow.
        location (str): The directory where the dataset will be downloaded.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_id)

    version_number = project.generate_version(settings={
        "augmentation":{},
        "preprocessing":{}
    })

    version = project.version(version_number)

    location = str(Path(location).absolute())
    logger.info(f"Downloading dataset to {location}")
    dataset = version.download(
        model_format='yolov8',
        location=location,
        overwrite=True
    )
    logger.info(f"Dataset downloaded to {dataset.location} successfully.")


def main():
    """
    Main function to parse arguments and download dataset.
    """
    # Load environment variables
    load_dotenv()

    today = datetime.today().strftime('%Y-%m-%d')

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow.")
    parser.add_argument("--api_key", default=os.getenv("API_KEY"), help="API key for Roboflow")
    parser.add_argument("--project_id", default=os.getenv("PROJECT_ID"), help="Project ID for Roboflow")
    parser.add_argument("--location", default=f"datasets/roboflow-raw/{today}", help="Location to save the dataset")
    args = parser.parse_args()

    # Download dataset
    download_dataset(args.api_key, args.project_id, args.location)

if __name__ == "__main__":
    main()

# Example usage:
# python scripts/download_roboflow.py --api_key YOUR_API_KEY --project_id YOUR_PROJECT_ID --location YOUR_LOCATION


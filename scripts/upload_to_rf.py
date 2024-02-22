import os
import argparse
import glob
from dotenv import load_dotenv
from roboflow import Roboflow

def main(workspace_id, project_id, image_files):
    load_dotenv()
    # Initialize the Roboflow object with your API key
    rf = Roboflow(api_key=os.environ.get('RF_PRIVATE_KEY'))

    # Retrieve your current workspace and project name
    print(rf.workspace())

    # Access the specified project
    project = rf.workspace(workspace_id).project(project_id)

    # Upload each image to your project
    for image_file in image_files:
        print(f"Uploading {image_file}...")
        project.upload(image_file)
        print(f"Uploaded {image_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload images to a Roboflow project.")
    parser.add_argument("--workspace_id", default="atom-scott-ix3vx", help="ID of the workspace")
    parser.add_argument("--project_id", default="soccertrack-v2-pbqrm", help="ID of the project within the workspace")
    parser.add_argument("image_files", nargs="+", help="Path to the image file(s) to upload")

    args = parser.parse_args()

    # Expand wildcard inputs
    expanded_files = []
    for file_pattern in args.image_files:
        expanded_files.extend(glob.glob(file_pattern))

    main(args.workspace_id, args.project_id, expanded_files)

import numpy as np
import argparse
from pathlib import Path
from sportslabkit.logger import logger
from sportslabkit.camera.calibrate import calibrate_video_from_mappings
from joblib import Parallel, delayed

def calibrate_video(video_path, mapx_path, mapy_path, save_path, overwrite):
    if not overwrite and save_path.exists():
        logger.info(f"Skipping existing file {save_path}")
        return

    # Load mapx and mapy
    mapx = np.load(mapx_path)
    mapy = np.load(mapy_path)

    # Calibrate camera from mappings
    calibrate_video_from_mappings(
        media_path=video_path,
        mapx=mapx,
        mapy=mapy,
        save_path=save_path
    )

def calibrate_videos_in_folder(input_folder, output_folder, n_jobs, overwrite):
    # Convert string paths to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Prepare list of tasks
    tasks = []
    for video_path in input_folder.glob('*.mp4'):
        mapx_path = input_folder / 'mapx.npy'
        mapy_path = input_folder / 'mapy.npy'

        if not mapx_path.exists() or not mapy_path.exists():
            logger.warning(f"Missing map files for video {video_path.name}")
            continue

        save_path = output_folder / video_path.name
        tasks.append((video_path, mapx_path, mapy_path, save_path, overwrite))

    # Process videos in parallel
    Parallel(n_jobs=n_jobs)(delayed(calibrate_video)(*task) for task in tasks)

def main():
    parser = argparse.ArgumentParser(description="Batch Calibrate Videos")
    parser.add_argument("--input_folder", required=True, type=str, help="Folder containing video and map files")
    parser.add_argument("--output_folder", required=True, type=str, help="Folder to save calibrated videos")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files in the output folder")
    args = parser.parse_args()

    calibrate_videos_in_folder(args.input_folder, args.output_folder, args.n_jobs, args.overwrite)

if __name__ == "__main__":
    main()

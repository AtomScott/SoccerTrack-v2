import typer
from omegaconf import OmegaConf
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
from tqdm import tqdm
import pandas as pd
from moviepy.editor import VideoFileClip
import numpy as np
import time
import tempfile
from joblib import Parallel, delayed
import yaml  # Ensure PyYAML is installed


app = typer.Typer()


def convert_video_fps(input_path: Path, output_path: Path, target_fps: int) -> None:
    """
    Convert a video to a specific FPS using MoviePy.

    Args:
        input_path (Path): Path to input video
        output_path (Path): Path to output video
        target_fps (int): Desired frames per second

    Raises:
        typer.Exit: If video conversion fails
    """
    try:
        with VideoFileClip(str(input_path)) as video_clip:
            video_clip_resized = video_clip.set_fps(target_fps)
            video_clip_resized.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
            )
    except Exception as e:
        logger.error(f"Failed to create temporary video {output_path}: {e}")
        raise typer.Exit(code=1)


def get_video_fps(video_path: Path) -> int:
    with VideoFileClip(str(video_path)) as video_clip:
        return video_clip.fps


def process_video(video_path: Path, config: dict, model: YOLO) -> None:
    """
    Process a single video: convert FPS, perform inference, and save results.

    Args:
        video_path (Path): Path to the input video.
        config (dict): Configuration dictionary.
        model (YOLO): Loaded YOLO model.
    """
    output_dir = Path(config["inference"]["output_dir"])
    output_csv_pattern = config["inference"].get("output_csv_pattern", "{base_name}.csv")
    tracker_config = config["inference"].get("tracker", {})
    logger.info(f"Using tracker config: {tracker_config}")

    fps = config["inference"]["fps"]
    conf = config["inference"]["conf"]
    iou = config["inference"]["iou"]
    imgsz = config["inference"]["imgsz"]

    base_name = video_path.stem
    output_csv = Path(output_dir) / output_csv_pattern.format(base_name=base_name)

    logger.info(f"Processing video: {video_path}")

    total_start_time = time.time()  # Start timing the entire process
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_video_path = Path(tmp_file.name)
            convert_video_fps(video_path, tmp_video_path, fps)

        # Create a temporary YAML file for tracker configuration
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_tracker_file:
            yaml.dump(tracker_config, tmp_tracker_file)
            tmp_tracker_path = Path(tmp_tracker_file.name)

        logger.info(f"Using tracker config: {tmp_tracker_path}")

        # Collect detections in a list
        detections_list = []
        inference_start_time = time.time()  # Start timing inference
        results = model.track(
            source=str(tmp_video_path),
            tracker=str(tmp_tracker_path),  # Pass the path to the temporary tracker config
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            stream=True,
        )

        frame_number = 0
        for res in tqdm(results, desc=f"Processing {base_name}"):
            boxes = res.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                track_id = int(box.id) if box.id is not None else -1
                conf_score = float(box.conf)
                cls = int(box.cls)

                mot_row = {
                    "frame": frame_number,
                    "id": track_id,
                    "bb_left": x1,
                    "bb_top": y1,
                    "bb_width": w,
                    "bb_height": h,
                    "conf": conf_score,
                    "x": -1,
                    "y": -1,
                    "z": -1,
                    "class": res.names[cls],
                }
                detections_list.append(mot_row)
            frame_number += 1

        # Create DataFrame
        detections_df = pd.DataFrame(detections_list)

        interpolated_dfs_list = []
        old_fps = get_video_fps(video_path)
        new_fps = fps
        for id, id_df in detections_df.groupby("id"):
            # Sort the DataFrame by frame and id
            id_df.sort_values(by=["frame", "id"], inplace=True)

            # Upsample the DataFrame to the new FPS
            max_frame = id_df["frame"].max()
            fps_ratio = old_fps / new_fps
            id_df["frame"] = id_df["frame"]

            new_frames = np.arange(0, max_frame * fps_ratio + 1)

            # Create a new DataFrame with the interpolated frames
            new_index = pd.Index(new_frames, name="frame")
            upsampled_df = id_df.set_index("frame")
            upsampled_df.index = upsampled_df.index * fps_ratio
            upsampled_df = upsampled_df.reindex(new_index)

            # Handle interpolation for bounding box coordinates
            bbox_cols = ["bb_left", "bb_top", "bb_width", "bb_height"]
            upsampled_df[bbox_cols] = upsampled_df[bbox_cols].interpolate(method="linear")

            # For IDs and classes, we can forward-fill and backward-fill, but avoid interpolation
            upsampled_df["id"] = upsampled_df["id"].fillna(method="ffill").fillna(method="bfill")
            upsampled_df["class"] = upsampled_df["class"].fillna(method="ffill").fillna(method="bfill")

            # Fill remaining NaNs in confidence with 0
            upsampled_df["conf"] = upsampled_df["conf"].fillna(0)

            # Set default values for columns x, y, z
            upsampled_df[["x", "y", "z"]] = upsampled_df[["x", "y", "z"]].fillna(-1)

            upsampled_df.reset_index(inplace=True)
            upsampled_df.dropna(inplace=True, axis=0)
            interpolated_dfs_list.append(upsampled_df)

        detections_df = pd.concat(interpolated_dfs_list)

        # Save DataFrame to CSV in MOT format
        # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
        mot_columns = [
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
            "class",
        ]
        detections_df = detections_df[mot_columns]
        detections_df["frame"] = detections_df["frame"].astype(int)
        detections_df["id"] = detections_df["id"].astype(int)
        detections_df.to_csv(output_csv, index=False, header=False)

        # Calculate timing metrics
        total_time = time.time() - total_start_time
        inference_time = time.time() - inference_start_time

        # Save timing metrics
        timing_df = pd.DataFrame(
            {
                "total_time": [total_time],
                "inference_time": [inference_time],
                "fps_conversion_and_post_processing": [total_time - inference_time],
            }
        )
        timing_df.to_csv(output_csv.with_suffix(".time"), index=False)

        logger.success(f"Processed {video_path}. Outputs saved to {output_csv}")
        logger.info(f"Total processing time: {total_time:.2f}s (Inference: {inference_time:.2f}s)")

    finally:
        # Clean up temporary files
        if tmp_video_path.exists():
            tmp_video_path.unlink()
            logger.debug(f"Cleaned up temporary file: {tmp_video_path}")
        if tmp_tracker_path.exists():
            tmp_tracker_path.unlink()
            logger.debug(f"Cleaned up temporary tracker config file: {tmp_tracker_path}")


@app.command()
def infer(
    config_path: str = typer.Option(..., help="Path to the YAML configuration file."),
):
    """
    Performs YOLO inference on videos specified in the configuration file.
    Generates annotated videos and detection results in CSV format in MOT format.
    """
    # Load the configuration file using OmegaConf
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise typer.Exit(code=1)

    # Convert OmegaConf object to a Python dictionary for easier access
    config = OmegaConf.to_container(config, resolve=True)
    logger.debug(f"Using config: {config}")
    logger.debug(f"Using tracker config: {config['inference']['tracker']}")

    # Extract inference parameters from the config
    model_weights = config["model"]["weights"]
    video_dir = config["inference"]["video_dir"]
    output_dir = config["inference"]["output_dir"]

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_weights)

    # Process each video file
    video_paths = list(Path(video_dir).glob("*.mp4"))
    if not video_paths:
        logger.error(f"No video files found in {video_dir}")
        raise typer.Exit(code=1)

    # Number of parallel jobs; adjust as needed or make it configurable
    num_jobs = config["inference"]["n_jobs"]

    # Use joblib to process videos in parallel
    Parallel(n_jobs=num_jobs)(delayed(process_video)(video_path, config, model) for video_path in video_paths)


if __name__ == "__main__":
    app()

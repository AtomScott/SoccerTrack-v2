# TrackNetのデータセットをYOLO形式に変換するコード
# change_tracknet_dataset_size.py実行後にできたtrain_100, train_500においては、名前をtrainに変更してから実行する

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import shutil
import sys

def load_numpy_file(file_path: Path):
    try:
        data = np.load(file_path, allow_pickle=True)
        logger.info(f"Loaded {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        sys.exit(1)

def create_yolo_label(class_id, x_center, y_center, width, height):
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_split(split_name, split_dir, output_dir, image_size):
    logger.info(f"Processing {split_name} split")
    
    frames_dir = split_dir / "frames"
    sequences_path = split_dir / "sequences.npy"
    coordinates_path = split_dir / "coordinates.npy"
    visibility_path = split_dir / "visibility.npy"
    
    # Load numpy files
    sequences = load_numpy_file(sequences_path)
    coordinates = load_numpy_file(coordinates_path)
    visibility = load_numpy_file(visibility_path)
    
    # Ensure the lengths match
    if not (len(sequences) == len(coordinates) == len(visibility)):
        logger.error("sequences.npy, coordinates.npy, and visibility.npy must have the same length")
        sys.exit(1)
    
    # Prepare output directories
    yolo_images_dir = output_dir / "images" / split_name
    yolo_labels_dir = output_dir / "labels" / split_name
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sequence
    for seq_idx in tqdm(range(len(sequences)), desc=f"Processing {split_name} sequences"):
        seq_frames = sequences[seq_idx]      # List of 3 frame paths
        seq_coords = coordinates[seq_idx]    # Shape: (3, 2)
        seq_visibility = visibility[seq_idx] # Shape: (3,)
        
        for frame_idx in range(len(seq_frames)):
            frame_path = Path(seq_frames[frame_idx])
            is_visible = seq_visibility[frame_idx]
            coord = seq_coords[frame_idx]
            
            # Define destination image path
            dest_image_path = yolo_images_dir / frame_path.name
            # Create a symbolic link if it doesn't exist
            if not dest_image_path.exists():
                try:
                    os.symlink(os.path.abspath(frame_path), dest_image_path)
                except Exception as e:
                    logger.error(f"Failed to create symlink for {frame_path} to {dest_image_path}: {e}")
                    continue
            
            # If the ball is visible, create a label
            if is_visible:
                x, y = coord  # Original coordinates in pixels
                box_width = 8
                box_height = 8
                
                # Normalize coordinates
                x_center_norm = x / image_size[0]
                y_center_norm = y / image_size[1]
                width_norm = box_width / image_size[0]
                height_norm = box_height / image_size[1]
                
                # Create label content
                label_content = create_yolo_label(0, x_center_norm, y_center_norm, width_norm, height_norm)
                
                # Define label file path
                label_filename = frame_path.stem + ".txt"
                label_file_path = yolo_labels_dir / label_filename
                
                # Write the label file
                try:
                    with open(label_file_path, "w") as f:
                        f.write(label_content + "\n")
                except Exception as e:
                    logger.error(f"Failed to write label file {label_file_path}: {e}")
                    continue
            else:
                # If not visible, ensure no label file exists
                label_file_path = yolo_labels_dir / (frame_path.stem + ".txt")
                if label_file_path.exists():
                    label_file_path.unlink()
    
    logger.info(f"Completed processing {split_name} split")

def main():
    # 定数の設定
    dataset_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset-stride-1")
    yolo_output_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset-stride-1_100frame")
    image_size = (3250, 500)  # [width, height] の指定
    
    # 出力ディレクトリ構造の作成
    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各分割データ（train, val, test）の処理
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist. Skipping.")
            continue
        process_split(split, split_dir, yolo_output_dir, image_size)
    
    logger.info("YOLO dataset creation completed successfully.")

if __name__ == "__main__":
    logger.remove()  # デフォルトのロガーを削除
    logger.add(sys.stderr, level="INFO")
    main()

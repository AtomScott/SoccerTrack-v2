"""Module for training YOLOv8 models."""

from pathlib import Path
from loguru import logger
from ultralytics import YOLO


def train_yolo_model(
    match_id: str,
    half: str,
    model_type: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    base_dir: Path | str | None = None,
    name: str = "soccer_player_detection",
    device: str = "0",
) -> None:
    """Train YOLOv8 model on soccer player dataset.

    Args:
        match_id: Match ID
        half: Which half of the match (1st or 2nd)
        model_type: YOLOv8 model type (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        base_dir: Base directory for data (defaults to data/interim/{match_id})
    """
    # Setup paths
    base_dir = Path(base_dir) if base_dir else Path(f"data/interim/{match_id}")
    dataset_dir = base_dir / f"ultralytics_format_{half}_half_distorted"
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    # Train model
    logger.info(f"Starting model training with {model_type}...")
    model = YOLO(model_type)  # load a pretrained model

    # Train the model with augmentations disabled
    model.train(
        data=str(dataset_dir / "dataset.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=50,  # Early stopping patience
        device=device,  # Use first GPU
        project="models",
        name=name,
        # Disable augmentations
        augment=False,  # No augmentations at all
        hsv_h=0.0,  # No hue shift
        hsv_s=0.0,  # No saturation shift
        hsv_v=0.0,  # No value shift
        degrees=0.0,  # No rotation
        translate=0.0,  # No translation
        scale=0.0,  # No scaling
        shear=0.0,  # No shearing
        perspective=0.0,  # No perspective
        flipud=0.0,  # No vertical flipping
        fliplr=0.0,  # No horizontal flipping
        bgr=0.0,  # Flip BGR color format
        mosaic=0.0,  # No mosaic
        mixup=0.0,  # No mixup
        copy_paste=0.0,  # No copy-paste
        erasing=0.0,  # No erasing
    )

    logger.success("Model training completed")

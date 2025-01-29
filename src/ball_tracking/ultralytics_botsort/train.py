"""Training script for Ultralytics YOLO model with tracking capabilities."""

from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO, settings
from ultralytics.utils import SETTINGS


def setup_ultralytics_settings(
    project_dir: Path,
    weights_dir: Path | None = None,
    datasets_dir: Path | None = None,
) -> None:
    """
    Configure Ultralytics settings for training.

    Args:
        project_dir: Root directory of the project.
        weights_dir: Directory to store model weights. If None, uses default.
        datasets_dir: Directory containing datasets. If None, uses default.
    """
    if weights_dir:
        settings.update({"weights_dir": str(weights_dir)})
    if datasets_dir:
        settings.update({"datasets_dir": str(datasets_dir)})

    # Update runs directory to be within our project
    settings.update({"runs_dir": str(project_dir / "models" / "runs")})
    logger.info(f"Updated Ultralytics settings: {SETTINGS}")


def train_yolo(
    data_yaml: Path,
    model_type: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str | None = None,
    project_name: str = "ball_tracking",
    name: str | None = None,
    resume: bool = False,
    pretrained: bool = True,
    optimizer: str = "auto",
    verbose: bool = True,
    **kwargs: Any,
) -> YOLO:
    """
    Train a YOLO model for object detection and tracking.

    Args:
        data_yaml: Path to the data configuration YAML file.
        model_type: Type of YOLO model to use (e.g., yolov8n.pt, yolov8s.pt).
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch_size: Training batch size.
        device: Device to train on ('cpu', '0', '0,1,2,3', etc.).
        project_name: Project name for organizing runs.
        name: Name of this specific run (auto-generated if None).
        resume: Resume training from last checkpoint.
        pretrained: Start with pretrained weights.
        optimizer: Optimizer to use (SGD, Adam, Adamax, etc.).
        verbose: Print verbose output.
        **kwargs: Additional arguments passed to model.train().

    Returns:
        YOLO: Trained YOLO model.

    Raises:
        FileNotFoundError: If data_yaml file doesn't exist.
        RuntimeError: If training fails.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")

    try:
        # Initialize model
        model = YOLO(model_type)
        if not pretrained:
            model = YOLO(model_type, pretrained=False)

        # Configure training parameters
        train_args = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch_size,
            "project": project_name,
            "name": name,
            "resume": resume,
            "optimizer": optimizer,
            "verbose": verbose,
            **kwargs,
        }

        if device:
            train_args["device"] = device

        # Start training
        logger.info(f"Starting training with parameters: {train_args}")
        model.train(**train_args)

        return model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}")


def train(cfg: DictConfig) -> None:
    """
    Main training function that uses configuration from OmegaConf.

    Args:
        cfg: Configuration object containing training parameters.
    """
    try:
        # Setup project directories
        project_dir = Path(__file__).parents[3]  # Go up 3 levels to project root
        setup_ultralytics_settings(project_dir)

        # Convert paths from config
        data_yaml = Path(cfg.ball_tracking.train.data_yaml)

        # Train model
        model = train_yolo(
            data_yaml=data_yaml,
            model_type=cfg.ball_tracking.train.model_type,
            epochs=cfg.ball_tracking.train.epochs,
            imgsz=cfg.ball_tracking.train.imgsz,
            batch_size=cfg.ball_tracking.train.batch_size,
            device=cfg.ball_tracking.train.device,
            project_name=cfg.ball_tracking.train.project_name,
            name=cfg.ball_tracking.train.name,
            resume=cfg.ball_tracking.train.resume,
            pretrained=cfg.ball_tracking.train.pretrained,
            optimizer=cfg.ball_tracking.train.optimizer,
        )

        # Save the final model
        final_model_path = project_dir / "models" / f"{cfg.ball_tracking.train.project_name}_final.pt"
        model.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Load config when running directly
    config_path = Path(__file__).parents[3] / "configs" / "default_config.yaml"
    cfg = OmegaConf.load(config_path)
    train(cfg)

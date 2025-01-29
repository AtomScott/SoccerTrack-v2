def create_yolo_dataset(
    video_path: Path = typer.Argument(
        ...,
        help="Path to the video file",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    mot_path: Path = typer.Argument(
        ...,
        help="Path to the MOT format annotation file",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for the dataset",
    ),
    frame_interval: int = typer.Option(
        5,
        help="Extract every Nth frame",
        min=1,
    ),
    train_split: float = typer.Option(
        0.8,
        help="Proportion of data for training",
        min=0.0,
        max=1.0,
    ),
    val_split: float = typer.Option(
        0.1,
        help="Proportion of data for validation",
        min=0.0,
        max=1.0,
    ),
    test_split: float = typer.Option(
        0.1,
        help="Proportion of data for testing",
        min=0.0,
        max=1.0,
    ),
    min_visibility: float = typer.Option(
        0.3,
        help="Minimum visibility score to include detection",
        min=0.0,
        max=1.0,
    ),
    frame_width: int = typer.Option(
        1280,
        help="Target frame width",
        min=1,
    ),
    frame_height: int = typer.Option(
        720,
        help="Target frame height",
        min=1,
    ),
    batch_size: int = typer.Option(
        32,
        help="Number of frames to process at once",
        min=1,
    ),
) -> Path:
    """Create a YOLO format dataset from a video and MOT format file."""
    try:
        if not abs(train_split + val_split + test_split - 1.0) < 1e-9:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Create dataset structure
        dataset_dir = output_dir / "dataset"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        
        # ... rest of the implementation ...
        # (keeping the core logic the same)

        logger.info(f"Dataset created successfully at {dataset_dir}")
        logger.info(f"Use {dataset_dir}/data.yaml for training")
        return dataset_dir
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(create_yolo_dataset) 
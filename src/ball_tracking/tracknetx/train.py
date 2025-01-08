import os
import time
import numpy as np
import sys
from pathlib import Path
from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from loguru import logger
from omegaconf import OmegaConf

from src.ball_tracking.tracknetx.data_module import TrackNetXDataModule
from src.ball_tracking.tracknetx.model import TrackNetXModel
from src.ball_tracking.tracknetx.utils import model_summary, evaluation, plot_result


def main():
    """Main training function."""
    # Load configuration
    config = OmegaConf.load("configs/default_config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    config = config.ball_tracking

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
    )

    # Initialize model
    model = TrackNetXModel(
        in_channels=config.train.in_channels,
        out_channels=config.train.out_channels,
        learning_rate=config.train.learning_rate,
        loss_function=config.train.main_loss,
        aux_loss_functions=config.train.auxiliary_losses,
        aux_loss_weights=config.train.auxiliary_weights,
    )

    # Initialize data module
    data_module = TrackNetXDataModule(
        root_dir=config.train.data_dir,
        num_frame=config.data.num_frame,
        stride=config.data.frame_stride,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        augmentation_config=config.data.augmentation,
        mag=config.data.mag,
        sigma=config.data.sigma,
        splits=list(config.data.splits.keys()),
    )

    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.train.save_dir,
            filename="model-{epoch:02d}-{val_total_loss:.2f}",
            save_top_k=3,
            monitor="val_total_loss",
            mode="min",
        ),
        EarlyStopping(
            monitor="val_total_loss",
            patience=config.train.early_stop_patience,
            mode="min",
        ),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator="auto",
        devices=config.train.devices,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    # Train the model
    logger.info("Starting training with validation...")
    trainer.fit(model, data_module)

    # Test the model
    logger.info("Starting testing...")
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()

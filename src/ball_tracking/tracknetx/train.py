import os
import argparse
import time
import numpy as np
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from tracknetx.data_module import TrackNetXDataModule
from tracknetx.model import TrackNetXModel
from tracknetx.utils import (
    model_summary,
    evaluation,
    plot_result,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the TrackNetX model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TrackNetV2",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--num_frame", type=int, default=3, help="Number of frames to use as input."
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="Input data type.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=4, help="Tolerance for evaluation metrics."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="exp",
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data and epochs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/TrackNetX_Dataset",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--devices", default=1, help="Number of GPUs to use for training."
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="TrackNetX",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name. Leave as None to use default.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name. If not set, it will be automatically generated.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal_wbce", "kl_div", "bce"],
        help="Main loss function to use.",
    )
    parser.add_argument(
        "--aux_loss_functions",
        type=str,
        default="tv",  # Comma-separated list of auxiliary loss functions
        help="Comma-separated list of auxiliary loss functions to use. Choices: tv, jaccard, dice",
    )
    parser.add_argument(
        "--aux_loss_weights",
        type=str,
        default="0.1",  # Comma-separated list of weights corresponding to auxiliary losses
        help="Comma-separated list of weights for the auxiliary loss functions.",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        default=640,
        help="Height of the random crop.",
    )
    parser.add_argument(
        "--crop_width",
        type=int,
        default=640,
        help="Width of the random crop.",
    )
    parser.add_argument(
        "--include_object_prob",
        type=float,
        default=0.8,
        help="Probability of including the object in the random crop.",
    )
    parser.add_argument(
        "--flip_prob",
        type=float,
        default=0.5,
        help="Probability of applying the horizontal flip.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=666,
        help="Target image height after resizing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=6500,
        help="Target image width after resizing.",
    )
    return parser.parse_args()


def main():
    """Main function for training the model."""
    args = parse_args()
    params = vars(args)
    # Configure logger
    os.makedirs(args.save_dir, exist_ok=True)
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level=args.log_level)
    logger.add(
        os.path.join(args.save_dir, "train.log"),
        rotation="500 MB",
        level=args.log_level,
    )
    logger.info(f"Training parameters: {params}")

    # Process auxiliary loss functions and weights
    if args.aux_loss_functions:
        aux_loss_functions = [
            loss.strip() for loss in args.aux_loss_functions.split(",")
        ]
    else:
        aux_loss_functions = []

    if args.aux_loss_weights:
        try:
            aux_loss_weights = [
                float(w.strip()) for w in args.aux_loss_weights.split(",")
            ]
            assert (
                len(aux_loss_weights) == len(aux_loss_functions)
            ), "Number of auxiliary loss weights must match the number of auxiliary loss functions."
        except:
            raise ValueError(
                "Auxiliary loss weights must be a comma-separated list of floats."
            )
    else:
        aux_loss_weights = [1.0] * len(aux_loss_functions)  # Default weight

    # Log the auxiliary losses being used
    logger.info(f"Auxiliary Loss Functions: {aux_loss_functions}")
    logger.info(f"Auxiliary Loss Weights: {aux_loss_weights}")

    # Initialize the DataModule
    data_module = TrackNetXDataModule(
        root_dir=args.data_dir,
        num_frame=args.num_frame,
        stride=1,
        batch_size=args.batch_size,
        num_workers=4 if not args.debug else 0,
        height=args.height,  # Height for resizing
        width=args.width,  # Width for resizing
        mag=1,
        sigma=2.5,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        include_object_prob=args.include_object_prob,  # Updated argument name
        flip_prob=args.flip_prob,
    )

    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup()

    # Initialize the model with auxiliary losses
    model = TrackNetXModel(
        in_channels=9,  # Adjust based on your input
        out_channels=3,  # Adjust based on your output
        learning_rate=args.learning_rate,  # Use the provided learning rate
        log_every_n_steps=50,
        loss_function=args.loss_function,
        aux_loss_functions=aux_loss_functions,
        aux_loss_weights=aux_loss_weights,
    )

    # Log model summary
    model_summary(model, args.model_name)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        filename="model_best",
        monitor="val_main_loss",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_main_loss",
        patience=args.early_stop_patience,
        verbose=True,
        mode="min",
    )

    # Initialize W&B Logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        save_dir=os.path.join(args.save_dir, "wandb_logs"),
        log_model=True,  # Optional: Log model checkpoints to W&B
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=[1],  # args.devices, #FIXME:
        accelerator="gpu",
        default_root_dir=args.save_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,  # Changed to W&B logger
        log_every_n_steps=50,
        limit_train_batches=0.02 if args.debug else 1.0,
        limit_val_batches=0.1 if args.debug else 1.0,
    )

    # Start training
    train_start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    total_training_time = (time.time() - train_start_time) / 3600.0
    logger.info(f"Training completed in {total_training_time:.2f} hours")

    # Evaluate the model
    logger.info("Starting evaluation on validation set...")
    accuracy, precision, recall, TP, TN, FP1, FP2, FN = evaluation(
        model, data_module.val_dataloader(), args.tolerance, args.input_type
    )
    logger.info(
        f"Validation Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
    )

    # Plot results
    plot_result(
        loss_list=None,  # Replace with actual loss history if available
        test_acc_dict={
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "TP": [len(TP)],
            "TN": [len(TN)],
            "FP1": [len(FP1)],
            "FP2": [len(FP2)],
            "FN": [len(FN)],
        },
    
        num_frame=args.num_frame,
        save_dir=args.save_dir,
        model_name=args.model_name,
    )

    logger.info("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()

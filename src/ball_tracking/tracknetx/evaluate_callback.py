# TrackNetの学習中の評価を出力するのに必要な関数

import pytorch_lightning as pl
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
import cv2
from tqdm import tqdm
import json

class EvaluateAndLogCallback(pl.Callback):
    def __init__(
        self,
        sequences,
        coordinates,
        visibility,
        model_checkpoint,
        device,
        distance_thresholds_list,  # List of threshold sets
        output_dir_list,  # Corresponding list of output directories
        draw_visualizations=False
    ):
        """
        Initialize the EvaluateAndLogCallback.

        Args:
            sequences (list): List of sequences, each sequence is a list of frame paths.
            coordinates (numpy.ndarray): Ground truth coordinates.
            visibility (numpy.ndarray): Visibility flags for each coordinate.
            model_checkpoint (str or None): Path to model checkpoint (unused here).
            device (str): Device to run evaluations on ("cpu" or "cuda").
            distance_thresholds_list (list of lists): List of distance threshold sets for mAP calculation.
            output_dir_list (list of str or Path): List of directories to save visualization images.
            draw_visualizations (bool): Whether to draw and save visualization images.
        """
        super().__init__()
        self.sequences = sequences
        self.coordinates = coordinates
        self.visibility = visibility
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.distance_thresholds_list = distance_thresholds_list
        self.output_dirs = [Path(dir) for dir in output_dir_list]
        for dir in self.output_dirs:
            dir.mkdir(parents=True, exist_ok=True)
        self.draw_visualizations = draw_visualizations

        # Initialize dictionaries to store metrics per threshold set
        self.metrics = {
            tuple(th): {'epochs': [], 'map_px': [], 'map_range': []}
            for th in self.distance_thresholds_list
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
            pl_module (pl.LightningModule): The model being trained.
        """
        # Set model to evaluation mode
        model = pl_module
        model.eval()
        model.to(self.device)

        current_epoch = trainer.current_epoch + 1  # Epoch count starts at 0

        for distance_thresholds, output_dir in zip(self.distance_thresholds_list, self.output_dirs):
            # Execute evaluation
            results = self.evaluate_tracknet_model(
                self.sequences,
                self.coordinates,
                self.visibility,
                model,
                output_dir,
                self.device,
                distance_thresholds
            )

            # Get the first threshold (assumed to be the largest due to sorting)
            main_thres = distance_thresholds[0]

            # Dynamically generate mAP keys
            map_px_key = f"mAP@{main_thres}px"
            map_range_key = f"mAP@{main_thres}~1px"

            # Check if the expected keys are present in the results
            if map_px_key not in results or map_range_key not in results:
                print(f"Warning: Expected keys {map_px_key} and {map_range_key} not found in results.")
                continue

            # Log metrics to WandB if using WandB logger
            if trainer.logger and isinstance(trainer.logger, pl.loggers.wandb.WandbLogger):
                wandb_logger = trainer.logger.experiment
                log_dict = {
                    "epoch": current_epoch,
                    map_px_key: results.get(map_px_key, 0.0),
                    map_range_key: results.get(map_range_key, 0.0)
                }

                # Additionally log individual mAP@{th}px for all thresholds
                for th, prec in results["individual_precisions"].items():
                    log_dict[f"mAP@{th}px"] = prec

                wandb_logger.log(log_dict)

            # Accumulate metrics for plotting
            th_tuple = tuple(distance_thresholds)
            self.metrics[th_tuple]['epochs'].append(current_epoch)
            self.metrics[th_tuple]['map_px'].append(results.get(map_px_key, 0.0))
            self.metrics[th_tuple]['map_range'].append(results.get(map_range_key, 0.0))

            # Print the mAP values
            print(f"Epoch {current_epoch}: {map_px_key} = {results[map_px_key]:.4f}, {map_range_key} = {results[map_range_key]:.4f}")

    def on_train_end(self, trainer, pl_module):
        """
        Called when the train ends. Saves the mAP over epochs plot.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
            pl_module (pl.LightningModule): The model being trained.
        """
        for distance_thresholds, output_dir in zip(self.distance_thresholds_list, self.output_dirs):
            th_tuple = tuple(distance_thresholds)
            metrics = self.metrics.get(th_tuple, None)
            if not metrics or not metrics['epochs']:
                print(f"No metrics collected for thresholds: {distance_thresholds}")
                continue

            main_thres = distance_thresholds[0]

            # Plot mAP over epochs
            plt.figure()
            plt.plot(metrics['epochs'], metrics['map_px'], label=f"mAP@{main_thres}px")
            plt.plot(metrics['epochs'], metrics['map_range'], label=f"mAP@{main_thres}~1px")
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title(f'mAP over Epochs (thres={main_thres}px)')
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = f"mAP_over_epochs_thres{main_thres}.png"
            plot_path = output_dir / plot_filename
            plt.savefig(str(plot_path))
            plt.close()

            print(f"Saved plot: {plot_filename} in {output_dir}")

            # Optionally, save results to a JSON file
            json_filename = f"evaluation_results_thres{main_thres}.json"
            json_path = output_dir / json_filename
            with open(json_path, 'w') as f:
                json.dump({
                    "epochs": metrics['epochs'],
                    "mAP@px": metrics['map_px'],
                    "mAP@~1px": metrics['map_range']
                }, f, indent=4)

            print(f"Saved evaluation results: {json_filename} in {output_dir}")

    ############################
    # 3) Utility Functions
    ############################

    def preprocess_frames(self, frames):
        """Preprocess frames for TrackNet input."""
        processed = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            processed.append(frame)
        input_tensor = np.concatenate(processed, axis=0)  # (C*3, H, W)
        return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

    def is_center_in_box(self, center, box_center, side_len=8):
        """Check if the predicted center is within the ground truth box."""
        half = side_len / 2
        bx1 = box_center[0] - half
        by1 = box_center[1] - half
        bx2 = box_center[0] + half
        by2 = box_center[1] + half
        return (bx1 <= center[0] <= bx2) and (by1 <= center[1] <= by2)

    def draw_predictions_and_ground_truth(self, img, pred_center, gt_center, detection_status, side_len=8):
        """
        Draw predicted and ground truth centers and boxes on the image.

        Args:
            img (numpy.ndarray): The image on which to draw.
            pred_center (tuple): Predicted (x, y) coordinates.
            gt_center (tuple): Ground truth (x, y) coordinates.
            detection_status (bool): Whether the detection was successful.
            side_len (int): Side length of the box to draw around centers.

        Returns:
            numpy.ndarray: The image with drawings.
        """
        half = side_len / 2
        # Define boxes
        pred_box = [
            int(pred_center[0] - half),
            int(pred_center[1] - half),
            int(pred_center[0] + half),
            int(pred_center[1] + half)
        ]
        gt_box = [
            int(gt_center[0] - half),
            int(gt_center[1] - half),
            int(gt_center[0] + half),
            int(gt_center[1] + half)
        ]

        # Draw rectangles
        cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)  # Green
        cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)  # Red

        # Draw centers
        cv2.circle(img, (int(pred_center[0]), int(pred_center[1])), 3, (0, 255, 0), -1)  # Green
        cv2.circle(img, (int(gt_center[0]), int(gt_center[1])), 3, (0, 0, 255), -1)  # Red

        # Prepare texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        x_text = 10
        y_text = 30
        y_offset = 30

        gt_text = f"Ground Truth: ({int(gt_center[0])}, {int(gt_center[1])})"
        pred_text = f"Prediction    : ({int(pred_center[0])}, {int(pred_center[1])})"
        detection_text = "Detection: Success" if detection_status else "Detection: Failure"

        # Put texts on image
        cv2.putText(img, gt_text, (x_text, y_text), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        cv2.putText(img, pred_text, (x_text, y_text + y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(img, detection_text, (x_text, y_text + 2 * y_offset), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

        return img

    ############################
    # 4) Evaluation Function
    ############################

    def evaluate_tracknet_model(
        self,
        sequences,
        coordinates,
        visibility,
        model,
        output_dir,
        device,
        distance_thresholds
    ):
        """
        Evaluate the TrackNet model on the validation set.

        Args:
            sequences (list): List of sequences, each sequence is a list of frame paths.
            coordinates (numpy.ndarray): Ground truth coordinates.
            visibility (numpy.ndarray): Visibility flags for each coordinate.
            model (pl.LightningModule): The model to evaluate.
            output_dir (str or Path): Directory to save visualization images.
            device (str): Device to run evaluations on ("cpu" or "cuda").
            distance_thresholds (list): List of distance thresholds for mAP calculation.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        distances = []
        distances_sq = []

        # Sort thresholds in descending order
        sorted_thresholds = sorted(distance_thresholds, reverse=True)
        dist_tp = {th: 0 for th in sorted_thresholds}
        dist_fp = {th: 0 for th in sorted_thresholds}
        individual_precisions = {}
        prec_values = []

        total_frames = len(sequences)

        for idx, sequence in enumerate(tqdm(sequences, total=total_frames, desc="Evaluating sequences")):
            # Load frames
            frames = [cv2.imread(str(frame_path)) for frame_path in sequence]
            if any(frame is None for frame in frames):
                print(f"Warning: Unable to read all frames in sequence {idx}. Skipping.")
                continue
            input_tensor = self.preprocess_frames(frames).to(device)

            # Inference
            with torch.no_grad():
                output = model(input_tensor)
                output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()

            # Get prediction from central heatmap
            pred_heatmap = output_prob[1]  # Assuming channel 1 is for prediction
            pred_y, pred_x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
            pred_center = (pred_x, pred_y)

            # Get ground truth
            gt_x, gt_y = coordinates[idx, 1]  # Assuming [sequence_index, frame_index, coordinates]
            gt_center = (gt_x, gt_y)
            vis = visibility[idx, 1]  # Assuming visibility is [sequence_index, frame_index]

            # Calculate distance
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            distances.append(dist)
            distances_sq.append(dist ** 2)

            # Update Precision/Recall
            if vis:
                if self.is_center_in_box(pred_center, gt_center, side_len=8):
                    true_positives += 1
                    detection_success = True
                else:
                    false_positives += 1
                    detection_success = False
            else:
                false_negatives += 1
                detection_success = False

            # Update mAP counters
            if vis:
                for th in sorted_thresholds:
                    if dist <= th:
                        dist_tp[th] += 1
                    else:
                        dist_fp[th] += 1

            # Draw and save visualizations if enabled
            if self.draw_visualizations:
                middle_frame_idx = len(sequence) // 2
                original_img = frames[middle_frame_idx].copy()
                vis_img = self.draw_predictions_and_ground_truth(
                    original_img,
                    pred_center,
                    gt_center,
                    detection_success,
                    side_len=8
                )
                frame_name = Path(sequence[middle_frame_idx]).stem
                save_path = output_dir / f"{frame_name}_vis.jpg"
                cv2.imwrite(str(save_path), vis_img)

        # Calculate Precision and Recall
        precision_center = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall_center = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        # Calculate mean distance and MSE
        mean_dist = np.mean(distances) if distances else 0.0
        mean_mse = np.mean(distances_sq) if distances_sq else 0.0

        # Calculate mAP for each threshold
        for th in sorted_thresholds:
            tp = dist_tp[th]
            fp = dist_fp[th]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            individual_precisions[th] = prec
            prec_values.append(prec)
            if th == sorted_thresholds[0]:
                main_map_key = f"mAP@{th}px"
                main_map_value = prec

        # Calculate mAP over range
        map_range_key = f"mAP@{sorted_thresholds[0]}~1px"
        map_range_value = np.mean(prec_values) if prec_values else 0.0

        # Compile results
        results = {
            "MeanEuclidianDistance": mean_dist,
            "MSE": mean_mse,
            "Precision_center": precision_center,
            "Recall_center": recall_center,
            main_map_key: main_map_value,
            map_range_key: map_range_value,
            "individual_precisions": individual_precisions
        }

        # Ensure all mAP@{th}px keys are present
        for th in sorted_thresholds:
            map_key = f"mAP@{th}px"
            if map_key not in results:
                results[map_key] = individual_precisions.get(th, 0.0)

        return results

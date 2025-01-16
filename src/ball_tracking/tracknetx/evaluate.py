import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, average_precision_score

from model import TrackNetXModel

def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load the trained model from a checkpoint."""
    model = TrackNetXModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model

def preprocess_frames(frames):
    """Preprocess frames to match the input format of the model."""
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        processed.append(frame)
    input_tensor = np.concatenate(processed, axis=0)  # (9, H, W)
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # (1, 9, H, W)

def draw_predictions(image, prediction, ground_truth, visibility, save_path):
    """
    Draw predictions, ground truth, and visibility on the image and save it.

    Args:
        image: Original image.
        prediction: Predicted (x, y) coordinates.
        ground_truth: Ground truth (x, y) coordinates.
        visibility: Whether the ball is visible in the frame.
        save_path: Path to save the annotated image.
    """
    img_with_overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Draw predicted position
    pred_x, pred_y = prediction
    cv2.circle(img_with_overlay, (int(pred_x), int(pred_y)), 5, (0, 255, 0), -1)  # Green circle
    cv2.putText(
        img_with_overlay,
        f"Pred: ({pred_x:.1f}, {pred_y:.1f})",
        (10, 30),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
    )

    # Draw ground truth position
    gt_x, gt_y = ground_truth
    cv2.circle(img_with_overlay, (int(gt_x), int(gt_y)), 5, (0, 0, 255), -1)  # Red circle
    cv2.putText(
        img_with_overlay,
        f"GT: ({gt_x:.1f}, {gt_y:.1f})",
        (10, 60),
        font,
        font_scale,
        (0, 0, 255),
        thickness,
    )

    # Draw visibility status
    vis_text = "Visible" if visibility else "Not Visible"
    vis_color = (0, 255, 0) if visibility else (0, 0, 255)
    cv2.putText(
        img_with_overlay,
        f"Visibility: {vis_text}",
        (10, 90),
        font,
        font_scale,
        vis_color,
        thickness,
    )

    # Save the image
    cv2.imwrite(str(save_path), img_with_overlay)

def evaluate_model(sequences, coordinates, visibility, model, output_dir, device):
    """Evaluate the model and calculate metrics."""
    predictions = []
    ground_truths = []
    visibilities = []
    ap_scores_50 = []
    ap_scores_50_95 = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for seq_idx, sequence in tqdm(enumerate(sequences), desc="Evaluating sequences"):
        frames = [cv2.imread(frame_path) for frame_path in sequence]
        input_tensor = preprocess_frames(frames).to(device)

        # Run prediction
        with torch.no_grad():
            output = model(input_tensor)
            output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()

        # Get predicted coordinates (argmax of heatmap)
        pred_heatmap = output_prob[1]  # Use the middle frame's heatmap
        pred_y, pred_x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        predictions.append((pred_x, pred_y))

        # Add ground truth and visibility
        ground_truths.append(tuple(coordinates[seq_idx, 1]))  # Middle frame's ground truth
        visibilities.append(visibility[seq_idx, 1])  # Middle frame's visibility

        # Prepare binary ground truth heatmap
        gt_heatmap = np.zeros_like(pred_heatmap)
        if visibility[seq_idx, 1]:  # Only mark visible frames
            gt_x, gt_y = coordinates[seq_idx, 1]
            gt_x, gt_y = int(gt_x), int(gt_y)
            if 0 <= gt_x < gt_heatmap.shape[1] and 0 <= gt_y < gt_heatmap.shape[0]:
                gt_heatmap[gt_y, gt_x] = 1  # Set ground truth pixel to 1

        # Flatten ground truth and predicted heatmaps for mAP calculation
        y_true = gt_heatmap.flatten()
        y_score = pred_heatmap.flatten()

        # Calculate AP for IoU thresholds
        ap_per_threshold = []
        for iou_thresh in iou_thresholds:
            ap_score = average_precision_score(y_true, y_score >= iou_thresh)
            ap_per_threshold.append(ap_score)
        ap_scores_50.append(ap_per_threshold[0])  # AP at IoU=0.5
        ap_scores_50_95.append(np.mean(ap_per_threshold))  # Mean AP from IoU=0.5 to 0.95

        # Draw and save prediction
        save_path = output_dir / f"sequence_{seq_idx}_frame_1.jpg"
        draw_predictions(frames[1], (pred_x, pred_y), coordinates[seq_idx, 1], visibility[seq_idx, 1], save_path)

    # Calculate overall metrics
    precision = precision_score(visibilities, [1 if v else 0 for v in visibilities])
    recall = recall_score(visibilities, [1 if v else 0 for v in visibilities])
    map_50 = np.mean(ap_scores_50) if ap_scores_50 else 0.0
    map_50_95 = np.mean(ap_scores_50_95) if ap_scores_50_95 else 0.0

    # Euclidean distance and MSE
    distances = []
    squared_errors = []
    for (pred_x, pred_y), (gt_x, gt_y), vis in zip(predictions, ground_truths, visibilities):
        if vis:  # Only evaluate visible frames
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            mse = (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2
            distances.append(dist)
            squared_errors.append(mse)

    mean_distance = np.mean(distances) if distances else float('nan')
    mean_squared_error = np.mean(squared_errors) if squared_errors else float('nan')

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_50_95,
        "mean_distance": mean_distance,
        "mean_squared_error": mean_squared_error,
    }

def main():
    # Paths
    sequences_path = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset/test/sequences.npy")
    coordinates_path = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset/test/coordinates.npy")
    visibility_path = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset/test/visibility.npy")
    checkpoint_path = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/tracknetx/exp-stride=3-weighted_msee/model-epoch=89-val_total_loss=0.00.ckpt"
    output_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/tracknetx/exp"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    sequences = np.load(sequences_path, allow_pickle=True)
    coordinates = np.load(coordinates_path, allow_pickle=True)
    visibility = np.load(visibility_path, allow_pickle=True)

    # Load model
    model = load_model(checkpoint_path, device)

    # Evaluate model
    metrics = evaluate_model(sequences, coordinates, visibility, model, output_dir, device)

    # Print metrics
    print("Evaluation Metrics:")
    print(f"Mean Euclidean Distance: {metrics['mean_distance']:.2f}")
    print(f"Mean Squared Error: {metrics['mean_squared_error']:.2f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")

if __name__ == "__main__":
    main()

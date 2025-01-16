import argparse
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

# Replace with the actual import of your TrackNetXModel
from model import TrackNetXModel


def load_model(checkpoint_path: str, device: str = "cpu"):
    model = TrackNetXModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def get_frame_triplet(frames_buffer, idx, total_frames):
    """
    Given a buffer of frames and an index, return a triplet of frames (prev, current, next).
    Handle boundary cases by replicating the first or last frame.
    """
    if idx == 0:
        # First frame, replicate it for "previous"
        return [frames_buffer[0], frames_buffer[0], frames_buffer[1]] if total_frames > 1 else [frames_buffer[0]] * 3
    elif idx == total_frames - 1:
        # Last frame, replicate it for "next"
        return (
            [frames_buffer[total_frames - 2], frames_buffer[total_frames - 1], frames_buffer[total_frames - 1]]
            if total_frames > 1
            else [frames_buffer[0]] * 3
        )
    else:
        # Middle frames
        return [frames_buffer[idx - 1], frames_buffer[idx], frames_buffer[idx + 1]]


def preprocess_frames_to_tensor(frames):
    """
    Preprocess a list of three frames (H,W,3 BGR):
    - Convert BGR to RGB
    - Normalize [0,1]
    - Stack into shape (C,H,W) with C=3*3=9 channels
    """
    processed = []
    for f in frames:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f_rgb = f_rgb.astype(np.float32) / 255.0
        f_rgb = f_rgb.transpose(2, 0, 1)  # (3,H,W)
        processed.append(f_rgb)
    # Stack along channel dimension: (3 frames * 3 channels) = 9 channels
    input_tensor = np.concatenate(processed, axis=0)  # (9,H,W)
    return torch.from_numpy(input_tensor).unsqueeze(0)  # (1,9,H,W)


def create_patch_coords(img_height, img_width, patch_size=480, overlap=80):
    """
    Create the top-left corners for extracting patches with given patch_size and overlap.
    Returns lists of y-coords and x-coords for patch positions.
    """

    def make_coords(dim, patch_size, overlap):
        coords = [0]
        step = patch_size - overlap
        while True:
            next_coord = coords[-1] + step
            if next_coord + patch_size >= dim:
                # last patch
                if coords[-1] == 0 and next_coord + patch_size > dim:
                    # If dimension is smaller than patch_size, just one patch
                    pass
                else:
                    coords.append(dim - patch_size)
                break
            else:
                coords.append(next_coord)
        return coords

    y_coords = make_coords(img_height, patch_size, overlap)
    x_coords = make_coords(img_width, patch_size, overlap)
    return y_coords, x_coords


def run_sliced_inference(model, input_tensor, device, patch_size=480, overlap=80, batch_size=None):
    """
    Run inference by splitting the input tensor into overlapping patches and processing all patches in a batch.
    """
    _, _, H, W = input_tensor.shape
    y_coords, x_coords = create_patch_coords(H, W, patch_size, overlap)

    # Extract patches
    patches = []
    patch_locations = []
    for y in y_coords:
        for x in x_coords:
            patch = input_tensor[:, :, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            patch_locations.append((y, x))

    # Stack all patches into a single batch
    all_patches = torch.cat(patches, dim=0).to(device)  # Shape: (N, 9, ph, pw)

    # If memory is a concern, you can split 'all_patches' into smaller chunks
    # For now, we assume we can process all at once.
    with torch.no_grad():
        outputs = model(all_patches)  # (N, C, ph, pw)
        outputs_probs = torch.sigmoid(outputs).cpu().numpy()

    out_c = model.out_channels
    final_map = np.zeros((out_c, H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Blend results back
    for (y, x), out_prob in zip(patch_locations, outputs_probs):
        ph, pw = out_prob.shape[1], out_prob.shape[2]
        final_map[:, y : y + ph, x : x + pw] += out_prob
        weight_map[y : y + ph, x : x + pw] += 1.0

    weight_map[weight_map == 0] = 1.0
    final_map = final_map / weight_map

    return final_map


def find_peak_coordinates_and_confidence(heatmap):
    """
    Given a heatmap (H,W), find peak coordinates and the confidence (max heatmap value).
    """
    flat_idx = np.argmax(heatmap)
    H, W = heatmap.shape
    y = flat_idx // W
    x = flat_idx % W
    confidence = heatmap[y, x]
    return x, y, confidence


def overlay_heatmap_on_frame(frame, heatmap, alpha=0.5):
    """
    Overlay a single-channel heatmap onto the original frame.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hm = heatmap
    if hm.max() > 0:
        hm = hm / hm.max()
    hm_color = np.zeros((hm.shape[0], hm.shape[1], 3), dtype=np.float32)
    hm_color[..., 0] = hm  # Red channel

    overlay = (1 - alpha) * frame_rgb + alpha * hm_color
    overlay = (overlay * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    return overlay_bgr


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps from video using sliced inference with temporal consistency."
    )
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video (mp4).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to output annotated video (mp4).")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV with predictions.")
    parser.add_argument("--patch_size", type=int, default=480, help="Patch size for sliced inference.")
    parser.add_argument("--overlap", type=int, default=80, help="Overlap size for sliced inference.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint_path, device=device)

    # Open input video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # Read all frames into memory (if memory allows)
    frames_buffer = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)
    cap.release()

    predictions = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)
    thickness = 2

    mid_idx = model.out_channels // 2

    heatmaps = []

    # Iterate over each frame index
    # We will do inference using frames [i-1, i, i+1] as input
    for i in tqdm(range(total_frames)):
        triplet = get_frame_triplet(frames_buffer, i, total_frames)
        # Preprocess these three frames
        input_tensor = preprocess_frames_to_tensor(triplet).to(device)  # shape: (1,9,H,W)

        # Run sliced inference
        patch_size = args.patch_size
        overlap = args.overlap
        outputs_probs = run_sliced_inference(model, input_tensor, device, patch_size, overlap)  # (C,H,W)
        heatmaps.append(outputs_probs)

        # Average over channels to get a single heatmap
        # If model.out_channels=3 and corresponds to the 3 input frames,
        final_heatmap = outputs_probs[mid_idx]  # (H,W)

        # Find peak
        x, y, confidence = find_peak_coordinates_and_confidence(final_heatmap)
        predictions.append([i, x, y, confidence])

        # Overlay heatmap
        overlay_frame = overlay_heatmap_on_frame(frames_buffer[i], final_heatmap)

        # Draw peak coordinate
        cv2.circle(overlay_frame, (x, y), 5, (0, 255, 0), -1)

        # Write to output video
        text = f"Frame: {i}, X: {x}, Y: {y}, Conf: {confidence:.2f}"
        cv2.putText(overlay_frame, text, (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
        out_video.write(overlay_frame)

    out_video.release()

    df = pd.DataFrame(predictions, columns=["frame", "x", "y", "confidence"])
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    print(f"Annotated video saved to {args.output_video}")


if __name__ == "__main__":
    main()
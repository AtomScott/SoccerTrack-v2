"""Analyze correlation between player positions and bounding box dimensions
using a fast grid-based template model with linear interpolation for smoothing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from loguru import logger
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
import joblib


def estimate_bbox_dimensions(
    x: float,
    y: float,
    width_model: pd.DataFrame,
    height_model: pd.DataFrame,
    ranges: dict | None = None,
) -> tuple[float, float]:
    """
    Estimate bounding box dimensions for a given position using the trained models.

    Args:
        x: X coordinate in image plane
        y: Y coordinate in image plane
        width_model: Tuple of (polynomial transform, regression model) for width prediction
        height_model: Linear regression model for height prediction
        ranges: Optional dictionary with data ranges for clamping predictions

    Returns:
        tuple[float, float]: Estimated (width, height) for the bounding box
    """
    poly, width_reg = width_model
    x_poly = poly.transform([[x]])
    estimated_width = width_reg.predict(x_poly)[0]
    estimated_height = height_model.predict([[y]])[0]

    # Clamp predictions to training data ranges if ranges are provided
    if ranges:
        estimated_width = np.clip(estimated_width, ranges["width_range"][0], ranges["width_range"][1])
        estimated_height = np.clip(estimated_height, ranges["height_range"][0], ranges["height_range"][1])

    return estimated_width, estimated_height


def load_bbox_models(model_path: Path | str) -> dict:
    """
    Load the saved grid-based template for bounding box dimensions.

    Args:
        model_path: Path to the saved model file

    Returns:
        dict: The grid template model dictionary.
    """
    return joblib.load(model_path)


def analyze_bbox_dimensions_fast(
    detections_path: Path | str,
    output_path: Path | str,
    match_id: str,
    conf_threshold: float = 0.3,
    grid_size: int = 20,
    interp_factor: int = 4,
) -> dict:
    """
    Analyze the correlation between player positions and their bounding box dimensions
    by binning detections into a grid, computing median dimensions per bin (ignoring bins with
    fewer than 10 samples), and then creating a fast, continuous interpolation model.

    Args:
        detections_path: Path to the detections CSV file.
        output_path: Path to save analysis files.
        match_id: Match ID for file naming.
        conf_threshold: Confidence threshold for filtering detections.
        grid_size: Number of bins for each dimension (x, y).
        interp_factor: Factor to refine the grid resolution for smoother visualization.

    Returns:
        dict: Dictionary containing:
              - 'x_edges': Bin edges for x.
              - 'y_edges': Bin edges for y.
              - 'width_grid': 2D array of median widths (bins with counts < 10 set to nan).
              - 'height_grid': 2D array of median heights (bins with counts < 10 set to nan).
              - 'ranges': dict with min/max info for x, y, width, height.
              - 'grid_size': The grid size used.
    """
    logger.info(f"Loading detections from {detections_path}")
    detections = pd.read_csv(
        detections_path,
        names=[
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
            "class_name",
        ],
    )

    # Filter by confidence and class "person"
    detections = detections[(detections["conf"] > conf_threshold) & (detections["class_name"] == "person")]
    print(detections.head())

    # Calculate center x and bottom y (player position)
    detections["center_x"] = detections["bb_left"] + detections["bb_width"] / 2
    detections["bottom_y"] = detections["bb_top"] + detections["bb_height"]

    # Create output directory (if needed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data ranges (useful for later clamping and for setting up the grid)
    model_info = {
        "x_range": (float(detections["center_x"].min()), float(detections["center_x"].max())),
        "y_range": (float(detections["bottom_y"].min()), float(detections["bottom_y"].max())),
        "width_range": (float(detections["bb_width"].min()), float(detections["bb_width"].max())),
        "height_range": (float(detections["bb_height"].min()), float(detections["bb_height"].max())),
    }

    # Log correlation statistics (optional)
    width_corr = stats.pearsonr(detections["center_x"], detections["bb_width"])
    height_corr = stats.pearsonr(detections["bottom_y"], detections["bb_height"])
    logger.info(f"Correlation between center_x and bb_width: r={width_corr[0]:.3f}, p={width_corr[1]:.3e}")
    logger.info(f"Correlation between bottom_y and bb_height: r={height_corr[0]:.3f}, p={height_corr[1]:.3e}")

    # --------------------------------------------------------------------------
    # Create a grid and accumulate sums and counts per bin
    # --------------------------------------------------------------------------
    x_min, x_max = model_info["x_range"]
    y_min, y_max = model_info["y_range"]
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    # Prepare accumulators for values in each bin
    width_values = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    height_values = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)

    # Determine bin indices for each detection (0-indexed)
    x_bin = np.digitize(detections["center_x"], x_edges) - 1
    y_bin = np.digitize(detections["bottom_y"], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)

    # Collect all values for each bin
    for i, (xb, yb) in enumerate(zip(x_bin, y_bin)):
        width_values[yb][xb].append(detections["bb_width"].iloc[i])
        height_values[yb][xb].append(detections["bb_height"].iloc[i])
        counts[yb, xb] += 1

    # Compute median for each bin with sufficient samples
    width_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    height_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    for i in range(grid_size):
        for j in range(grid_size):
            if counts[i, j] >= 10:
                width_grid[i, j] = np.median(width_values[i][j])
                height_grid[i, j] = np.median(height_values[i][j])
            else:
                width_grid[i, j] = np.nan
                height_grid[i, j] = np.nan

    # --------------------------------------------------------------------------
    # Use a fast linear interpolation on the binned medians for smooth output.
    # --------------------------------------------------------------------------
    # Get bin centers from edges
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create a high-resolution grid for visualization
    x_interp = np.linspace(x_edges[0], x_edges[-1], grid_size * interp_factor)
    y_interp = np.linspace(y_edges[0], y_edges[-1], grid_size * interp_factor)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

    # Fill missing (nan) values with the overall median (for interpolation purposes)
    width_filled = np.nan_to_num(width_grid, nan=np.nanmedian(width_grid))
    height_filled = np.nan_to_num(height_grid, nan=np.nanmedian(height_grid))

    # Build fast linear interpolators
    width_interp = RegularGridInterpolator((y_centers, x_centers), width_filled, method="slinear", bounds_error=False)
    height_interp = RegularGridInterpolator((y_centers, x_centers), height_filled, method="slinear", bounds_error=False)

    # Prepare points for interpolation on the high-resolution grid
    points = np.column_stack((Y_interp.ravel(), X_interp.ravel()))
    width_smooth = width_interp(points).reshape(X_interp.shape)
    height_smooth = height_interp(points).reshape(X_interp.shape)

    # --------------------------------------------------------------------------
    # Visualize the interpolated heatmaps
    # --------------------------------------------------------------------------
    # Width heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(X_interp, Y_interp, width_smooth, shading="auto", cmap="viridis")
    ax.set_title("Bounding Box Width Heatmap (Median-based, Interpolated)")
    ax.set_xlabel("X Position (Center)")
    ax.set_ylabel("Y Position (Bottom)")
    fig.colorbar(c, ax=ax, label="Width (pixels)")
    ax.grid(which="both", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    plt.savefig(output_path.with_suffix(".width_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Height heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(X_interp, Y_interp, height_smooth, shading="auto", cmap="plasma")
    ax.set_title("Bounding Box Height Heatmap (Median-based, Interpolated)")
    ax.set_xlabel("X Position (Center)")
    ax.set_ylabel("Y Position (Bottom)")
    fig.colorbar(c, ax=ax, label="Height (pixels)")
    ax.grid(which="both", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    plt.savefig(output_path.with_suffix(".height_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Log outputs
    logger.info(f"Total of {counts.sum()} detections processed for grid template")
    logger.info(f"Output saved to {output_path.with_suffix('.width_heatmap.png')}")
    logger.info(f"Output saved to {output_path.with_suffix('.height_heatmap.png')}")

    # Package grid template and model information for later use
    grid_dict = {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "width_grid": width_grid,
        "height_grid": height_grid,
        "ranges": model_info,
        "grid_size": grid_size,
    }

    logger.info("Saving fast grid-based template model and data ranges...")
    joblib.dump(grid_dict, output_path)

    return grid_dict


def estimate_bbox_dimensions_fast(
    x: float,
    y: float,
    grid_dict: dict,
    ranges: dict | None = None,
    mode: str = "bilinear",
) -> tuple[float, float]:
    """
    Estimate bounding box dimensions for a given (x, y) position using the fast grid-based model.

    Args:
        x: X coordinate in the image plane.
        y: Y coordinate in the image plane.
        grid_dict: Dictionary from the saved grid model.
        ranges: Optional dictionary with data ranges for clamping predictions.
        mode: "nearest" (default cell value) or "bilinear" interpolation.

    Returns:
        tuple[float, float]: Estimated (width, height) for the bounding box.
    """
    x_edges = grid_dict["x_edges"]
    y_edges = grid_dict["y_edges"]
    width_grid = grid_dict["width_grid"]
    height_grid = grid_dict["height_grid"]

    # Find bin indices
    ix = np.searchsorted(x_edges, x) - 1
    iy = np.searchsorted(y_edges, y) - 1
    ix = np.clip(ix, 0, len(x_edges) - 2)
    iy = np.clip(iy, 0, len(y_edges) - 2)

    if mode == "nearest":
        w_est = width_grid[iy, ix]
        h_est = height_grid[iy, ix]
        # Fallback to overall mean if the bin has insufficient data
        if np.isnan(w_est) or np.isnan(h_est):
            w_est = np.nanmean(width_grid)
            h_est = np.nanmean(height_grid)
    elif mode == "bilinear":
        # Bilinear interpolation using the four surrounding bin averages
        x_left, x_right = x_edges[ix], x_edges[ix + 1]
        y_bottom, y_top = y_edges[iy], y_edges[iy + 1]
        dx = (x - x_left) / (x_right - x_left) if x_right != x_left else 0.0
        dy = (y - y_bottom) / (y_top - y_bottom) if y_top != y_bottom else 0.0

        Q11w = width_grid[iy, ix]
        Q21w = width_grid[iy, ix + 1] if ix + 1 < width_grid.shape[1] else Q11w
        Q12w = width_grid[iy + 1, ix] if iy + 1 < width_grid.shape[0] else Q11w
        Q22w = width_grid[iy + 1, ix + 1] if (iy + 1 < width_grid.shape[0] and ix + 1 < width_grid.shape[1]) else Q11w
        w_est = Q11w * (1 - dx) * (1 - dy) + Q21w * dx * (1 - dy) + Q12w * (1 - dx) * dy + Q22w * dx * dy

        Q11h = height_grid[iy, ix]
        Q21h = height_grid[iy, ix + 1] if ix + 1 < height_grid.shape[1] else Q11h
        Q12h = height_grid[iy + 1, ix] if iy + 1 < height_grid.shape[0] else Q11h
        Q22h = (
            height_grid[iy + 1, ix + 1] if (iy + 1 < height_grid.shape[0] and ix + 1 < height_grid.shape[1]) else Q11h
        )
        h_est = Q11h * (1 - dx) * (1 - dy) + Q21h * dx * (1 - dy) + Q12h * (1 - dx) * dy + Q22h * dx * dy

        # Fallback if the interpolation returns nan
        if np.isnan(w_est):
            w_est = np.nanmean(width_grid)
        if np.isnan(h_est):
            h_est = np.nanmean(height_grid)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'nearest' or 'bilinear'.")

    if ranges:
        w_est = np.clip(w_est, ranges["width_range"][0], ranges["width_range"][1])
        h_est = np.clip(h_est, ranges["height_range"][0], ranges["height_range"][1])

    return float(w_est), float(h_est)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze bounding box dimensions using a fast grid-based template model"
    )
    parser.add_argument("--detections_path", type=str, required=True, help="Path to detections CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save analysis files")
    parser.add_argument("--match_id", type=str, required=True, help="Match ID for file naming")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.3, help="Confidence threshold for filtering detections"
    )
    parser.add_argument(
        "--grid_size", type=int, default=20, help="Number of bins per dimension for grid-based template"
    )
    parser.add_argument("--interp_factor", type=int, default=4, help="Interpolation factor for smooth visualization")

    args = parser.parse_args()

    grid_dict = analyze_bbox_dimensions_fast(
        detections_path=args.detections_path,
        output_path=args.output_path,
        match_id=args.match_id,
        conf_threshold=args.conf_threshold,
        grid_size=args.grid_size,
        interp_factor=args.interp_factor,
    )

    # Example usage of the estimation function
    test_x, test_y = 500, 300  # Example coordinates
    saved_grid_dict = joblib.load(args.output_path)
    est_width, est_height = estimate_bbox_dimensions_fast(
        x=test_x,
        y=test_y,
        grid_dict=saved_grid_dict,
        ranges=saved_grid_dict["ranges"],
        mode="bilinear",  # "nearest" or "bilinear"
    )
    logger.info(
        f"For position ({test_x}, {test_y}), estimated dimensions: width={est_width:.1f}, height={est_height:.1f}"
    )

"""Analyze correlation between player positions and bounding box dimensions."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib


def analyze_bbox_dimensions(
    detections_path: Path | str,
    output_path: Path | str,
    match_id: str,
    conf_threshold: float = 0.3,
) -> tuple[tuple[PolynomialFeatures, LinearRegression], LinearRegression]:
    """
    Analyze the correlation between player positions and their bounding box dimensions.
    Creates scatter plots and fits regression models.
    Uses 2nd degree polynomial for width vs x, and linear for height vs y.

    Args:
        detections_path: Path to the detections CSV file
        output_path: Path to save analysis files
        match_id: Match ID for file naming
        conf_threshold: Confidence threshold for filtering detections

    Returns:
        tuple[tuple[PolynomialFeatures, LinearRegression], LinearRegression]:
            Models for width (poly transform + regression) and height prediction
    """
    logger.info(f"Loading detections from {detections_path}")
    detections = pd.read_csv(
        detections_path,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z", "class_name"],
    )

    # Filter by confidence and class
    detections = detections[(detections["conf"] > conf_threshold) & (detections["class_name"] == "person")]

    # Calculate center x and bottom y
    detections["center_x"] = detections["bb_left"] + detections["bb_width"] / 2
    detections["bottom_y"] = detections["bb_top"] + detections["bb_height"]

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the data ranges for normalization during prediction
    model_info = {
        "x_range": (float(detections["center_x"].min()), float(detections["center_x"].max())),
        "y_range": (float(detections["bottom_y"].min()), float(detections["bottom_y"].max())),
        "width_range": (float(detections["bb_width"].min()), float(detections["bb_width"].max())),
        "height_range": (float(detections["bb_height"].min()), float(detections["bb_height"].max())),
    }

    # # Analyze width correlation with polynomial
    # plt.figure(figsize=(10, 6))
    # plt.scatter(detections["center_x"], detections["bb_width"], alpha=0.1)
    # plt.xlabel("X Position")
    # plt.ylabel("Bounding Box Width")
    # plt.title("X Position vs Bounding Box Width")

    # # Calculate correlation
    # width_correlation = stats.pearsonr(detections["center_x"], detections["bb_width"])
    # plt.text(
    #     0.05,
    #     0.95,
    #     f"Correlation: {width_correlation[0]:.3f}\np-value: {width_correlation[1]:.3e}",
    #     transform=plt.gca().transAxes,
    # )

    # plt.savefig(output_path.with_suffix(".width_correlation.png"))
    # plt.close()

    # # Analyze height correlation
    # plt.figure(figsize=(10, 6))
    # plt.scatter(detections["bottom_y"], detections["bb_height"], alpha=0.1)
    # plt.xlabel("Y Position (Bottom)")
    # plt.ylabel("Bounding Box Height")
    # plt.title("Y Position vs Bounding Box Height")

    # # Calculate correlation
    # height_correlation = stats.pearsonr(detections["bottom_y"], detections["bb_height"])
    # plt.text(
    #     0.05,
    #     0.95,
    #     f"Correlation: {height_correlation[0]:.3f}\np-value: {height_correlation[1]:.3e}",
    #     transform=plt.gca().transAxes,
    # )

    # plt.savefig(output_path.with_suffix(".height_correlation.png"))
    # plt.close()

    # Fit regression models
    # For width model - polynomial
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_width_poly = poly.fit_transform(detections[["center_x"]].values)
    width_model = LinearRegression()
    width_model.fit(X_width_poly, detections["bb_width"].values)

    # For height model - linear
    height_model = LinearRegression()
    X_height = detections[["bottom_y"]].values
    y_height = detections["bb_height"].values
    height_model.fit(X_height, y_height)

    # Plot regression lines
    plt.figure(figsize=(10, 6))
    plt.scatter(detections["center_x"], detections["bb_width"], alpha=0.1)
    x_range = np.linspace(detections["center_x"].min(), detections["center_x"].max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    plt.plot(x_range, width_model.predict(x_range_poly), "r-", label="Polynomial Regression (degree=2)")
    plt.xlabel("X Position")
    plt.ylabel("Bounding Box Width")
    plt.title("Width Regression Model")
    plt.text(
        0.05,
        0.95,
        f"R² Score: {width_model.score(X_width_poly, detections['bb_width']):.3f}",
        transform=plt.gca().transAxes,
    )
    plt.legend()
    plt.savefig(output_path.with_suffix(".width_regression.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(detections["bottom_y"], detections["bb_height"], alpha=0.1)
    y_range = np.linspace(detections["bottom_y"].min(), detections["bottom_y"].max(), 100).reshape(-1, 1)
    plt.plot(y_range, height_model.predict(y_range), "r-", label="Linear Regression")
    plt.xlabel("Y Position (Bottom)")
    plt.ylabel("Bounding Box Height")
    plt.title("Height Regression Model")
    plt.text(0.05, 0.95, f"R² Score: {height_model.score(X_height, y_height):.3f}", transform=plt.gca().transAxes)
    plt.legend()
    plt.savefig(output_path.with_suffix(".height_regression.png"))
    plt.close()

    # Print model coefficients
    logger.info(
        f"Width model (polynomial): y = {width_model.coef_[1]:.3f}x² + {width_model.coef_[0]:.3f}x + {width_model.intercept_:.3f}"
    )
    logger.info(f"Height model (linear): y = {height_model.coef_[0]:.3f}x + {height_model.intercept_:.3f}")

    # Save the models and ranges
    logger.info("Saving regression models and data ranges")
    joblib.dump(
        {"width_poly": poly, "width_model": width_model, "height_model": height_model, "ranges": model_info},
        output_path,
    )

    return (poly, width_model), height_model


def load_bbox_models(
    model_path: Path | str,
) -> tuple[tuple[PolynomialFeatures, LinearRegression], LinearRegression, dict]:
    """
    Load the saved bbox regression models and data ranges.

    Args:
        model_path: Path to the saved models file

    Returns:
        tuple[tuple[PolynomialFeatures, LinearRegression], LinearRegression, dict]:
            Width model (poly transform + regression), height model, and data ranges
    """
    models_dict = joblib.load(model_path)
    return ((models_dict["width_poly"], models_dict["width_model"]), models_dict["height_model"], models_dict["ranges"])


def estimate_bbox_dimensions(
    x: float,
    y: float,
    width_model: tuple[PolynomialFeatures, LinearRegression],
    height_model: LinearRegression,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bounding box dimensions correlation with position")
    parser.add_argument("--detections_path", type=str, required=True, help="Path to detections CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save analysis files")
    parser.add_argument("--match_id", type=str, required=True, help="Match ID for file naming")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.3, help="Confidence threshold for filtering detections"
    )

    args = parser.parse_args()

    width_model, height_model = analyze_bbox_dimensions(
        detections_path=args.detections_path,
        output_path=args.output_path,
        match_id=args.match_id,
        conf_threshold=args.conf_threshold,
    )

    # Example usage of the estimation function
    test_x, test_y = 500, 300  # Example coordinates
    est_width, est_height = estimate_bbox_dimensions(test_x, test_y, width_model, height_model)
    logger.info(
        f"For position ({test_x}, {test_y}), estimated dimensions: width={est_width:.1f}, height={est_height:.1f}"
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert annotated pitch keypoints JSON (pitch coords -> *distorted* image pixels)
into a JSON whose values are the *calibrated* (undistorted) image pixels,
by applying a pitch->image homography (3x3).

Input JSON format (example):
{
  "(0,68)": [111.399, 598.321],
  "(5.5,68)": [161.899, 605.488],
  ...
}
* Keys are pitch-plane coordinates in meters. By default, we assume a CORNER-origin
  coordinate (x in [0,105], y in [0,68]).
* Values are ignored for the calibrated conversion; we recompute from pitch coords.

Homography:
- We expect an npy file containing a 3x3 homography that maps from pitch-plane
  coordinates (meters) to *calibrated* image-plane pixels.

If your pitch coords are CENTER-origin (x in [-52.5,52.5], y in [-34,34]),
use --coords_frame center (we will shift to corner-origin before applying H).

Usage example:
    python convert_keypoints_json_to_calibrated.py \
        --input_json /data/share/SoccerTrack-v2/data/raw/117093/117093_keypoints.json \
        --homography_npy data/interim/homography/117093/117093_homography.npy \
        --output_json /data/share/SoccerTrack-v2/data/raw/117093/117093_keypoints_calibrated.json \
        --coords_frame corner \
        --pitch_length 105.0 \
        --pitch_width 68.0
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import cv2


def parse_world_key(s: str) -> Tuple[float, float]:
    """Parse a string like '(52.5,68)' -> (52.5, 68.0)."""
    m = re.match(r"\(([-\d\.]+)\s*,\s*([-\d\.]+)\)", s.strip())
    if not m:
        raise ValueError(f"Invalid key format: {s}")
    return float(m.group(1)), float(m.group(2))


def build_pitch_points_array(keys: List[str],
                             coords_frame: str,
                             pitch_length: float,
                             pitch_width: float) -> np.ndarray:
    """
    Build an (N,1,2) float32 array of pitch-plane points in the coordinate
    frame expected by the homography.

    coords_frame:
      - 'corner': keys already are (x in [0,105], y in [0,68]) => use as-is
      - 'center': keys are (x in [-52.5,52.5], y in [-34,34])  => shift to corner by (+52.5, +34)
    """
    pts = []
    for k in keys:
        xw, yw = parse_world_key(k)
        if coords_frame == "center":
            xw = xw + pitch_length / 2.0  # shift +52.5
            yw = yw + pitch_width  / 2.0  # shift +34
        # if 'corner', use as-is
        pts.append([xw, yw])

    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return pts


def apply_homography(pts_pitch: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 homography to (N,1,2) pitch-plane points -> (N,2) image pixels.
    """
    img_pts = cv2.perspectiveTransform(pts_pitch, H)   # (N,1,2)
    img_pts = img_pts.reshape(-1, 2)                   # (N,2)
    return img_pts


def convert_keypoints_json_to_calibrated(
    input_json: Path | str,
    homography_npy: Path | str,
    output_json: Path | str,
    coords_frame: str = "corner",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    rounding: int | None = None,
) -> Path:
    """
    Main converter:
      - Load JSON (keys are pitch coords, values ignored)
      - Build pitch points array (meters)
      - Apply H (pitch->image calibrated)
      - Write JSON with same keys and calibrated pixels as values
    """
    input_json = Path(input_json)
    homography_npy = Path(homography_npy)
    output_json = Path(output_json)

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")
    if not homography_npy.exists():
        raise FileNotFoundError(f"Homography npy not found: {homography_npy}")

    with open(input_json, "r") as f:
        data: Dict[str, List[float]] = json.load(f)

    keys = list(data.keys())

    # Build pitch-plane points compatible with H
    pts_pitch = build_pitch_points_array(
        keys,
        coords_frame=coords_frame,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    # Load H (3x3)
    H = np.load(homography_npy)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")

    # Apply H to get *calibrated* image pixels
    pts_img = apply_homography(pts_pitch, H)  # (N,2)

    # Build output dict with same keys, new values
    out_dict: Dict[str, List[float]] = {}
    for k, (u, v) in zip(keys, pts_img):
        if rounding is not None:
            u = round(float(u), rounding)
            v = round(float(v), rounding)
        else:
            u = float(u)
            v = float(v)
        out_dict[k] = [u, v]

    # Save
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    return output_json


def main():
    ap = argparse.ArgumentParser(description="Convert pitch keypoints JSON to calibrated image pixels via homography.")
    ap.add_argument("--input_json", required=True, help="Path to input keypoints JSON (pitch coords : distorted pixels).")
    ap.add_argument("--homography_npy", required=True, help="Path to 3x3 homography .npy (pitch->calibrated image).")
    ap.add_argument("--output_json", required=True, help="Where to save calibrated keypoints JSON.")
    ap.add_argument("--coords_frame", choices=["corner", "center"], default="corner",
                    help="Frame of pitch coords in JSON keys. 'corner' = (0..105,0..68), 'center' = (-52.5..52.5,-34..34).")
    ap.add_argument("--pitch_length", type=float, default=105.0, help="Pitch length (m).")
    ap.add_argument("--pitch_width",  type=float, default=68.0,  help="Pitch width (m).")
    ap.add_argument("--round", dest="rounding", type=int, default=2, help="Round pixel outputs to N decimals (default 2).")

    args = ap.parse_args()
    out = convert_keypoints_json_to_calibrated(
        input_json=args.input_json,
        homography_npy=args.homography_npy,
        output_json=args.output_json,
        coords_frame=args.coords_frame,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        rounding=args.rounding,
    )
    print(f"[OK] wrote: {out}")


if __name__ == "__main__":
    main()

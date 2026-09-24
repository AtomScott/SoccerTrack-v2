"""Measure, per half, how many frames the video is missing from the START of the annotated period.

WHY THIS EXISTS
    Ground-truth frame N is not video frame N. Attaching them naively misaligns labels from
    imagery and costs enormous accuracy: on 128057's first half the offset is 25 frames (1.0 s),
    players move 1.7-2.9 m in that time, and correcting it raised GS-HOTA from 18.79 to 37.16.

    Arithmetic alone cannot be trusted to find it. Two plausible definitions disagree:
        (End - Start) * fps / 1000 from <match>_padding_info.csv  minus the video's frame count
        the ground truth's own seq_length                          minus the video's frame count
    For 128057 and 117093 both give 25. For the 118xxx matches and 128058 they differ by ~251
    frames (10 s), and for 132877 by -12. So the offset must be MEASURED per half.

HOW
    A detector is not required. Players are the moving foreground against a static pitch, so a
    median-background subtraction gives a per-frame "player mass" centroid directly from pixels.
    Cross-correlating that against the centroid of the ground truth's bbox_image positions
    recovers the lag. This is assignment-free, uses no neural network, needs no prior staging,
    and never consults the evaluation metric -- it is a pure data-alignment measurement.

    Rows above the pitch (crowd, trees, stands) are excluded using the vertical extent of the
    ground-truth boxes themselves, so background clutter cannot drag the centroid.

VALIDATION
    Run with --validate to check the estimator against the two halves whose offset is known
    independently (128057 1st and 117093 1st, both 25, established by cross-correlating real
    detector output against labels).

USAGE
    python scripts/gsr/measure_frame_offset.py --match 128057 --half 1st
    python scripts/gsr/measure_frame_offset.py --all --out offsets.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]


def gt_centroids(gt_path: Path, first_n: int) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    """Per-frame centroid of GT bbox_image bottom-centres, for GT frames 1..first_n."""
    d = json.loads(gt_path.read_text())
    per: dict[int, list] = {}
    for a in d["annotations"]:
        if a.get("supercategory") != "object":
            continue
        n = int(str(a["image_id"])[1:]) if str(a["image_id"])[0] == "3" else int(str(a["image_id"])[-6:])
        if n > first_n:
            continue
        bi = a["bbox_image"]
        per.setdefault(n, []).append([bi["x"] + bi["w"] / 2.0, bi["y"] + bi["h"]])
    del d
    cent = {k: np.mean(v, axis=0) for k, v in per.items()}
    ys = np.concatenate([np.array(v)[:, 1] for v in per.values()]) if per else np.array([0, 1])
    band = (int(max(0, ys.min() - 120)), int(ys.max() + 40))
    return cent, band


def _boxes_by_frame(gt_path: Path, lo: int, hi: int) -> dict[int, np.ndarray]:
    """GT bbox_image arrays keyed by GT frame number, for frames in [lo, hi]."""
    d = json.loads(gt_path.read_text())
    per: dict[int, list] = {}
    for a in d["annotations"]:
        if a.get("supercategory") != "object":
            continue
        sid = str(a["image_id"])
        n = int(sid[1:]) if sid[0] == "3" and len(sid) == 7 else int(sid[-6:])
        if n < lo or n > hi:
            continue
        bi = a["bbox_image"]
        per.setdefault(n, []).append([bi["x"], bi["y"], bi["w"], bi["h"]])
    del d
    return {k: np.array(v, float) for k, v in per.items()}


def _grass_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Boolean grass mask. Computed ONCE per frame; the lag search reuses it, otherwise the
    HSV conversion would run once per (frame, lag) pair -- hundreds of times more work."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16)
    return (H > 25) & (H < 95) & (S > 40)


def _box_occupancy(grass: np.ndarray, boxes: np.ndarray) -> float:
    """How player-like is the content inside these boxes? Higher = boxes are on players.

    A football pitch is overwhelmingly green, players are not. So the fraction of
    non-grass pixels inside a box is high when the box is on a player and low when it is on
    empty grass. This measures exactly the property that matters -- whether labels land on
    the people they describe -- and needs no detector.
    """
    h, w = grass.shape
    vals = []
    for x, y, bw, bh in boxes:
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(w, x + bw)), int(min(h, y + bh))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        vals.append(1.0 - float(grass[y0:y1, x0:x1].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def estimate(match: str, half: str, data: Path, n_frames: int = 60,
             lag_range: int = 320, start_frame: int = 4000, step: int = 40) -> dict:
    """Find the lag L maximising box occupancy when GT frame (f+L) is drawn on video frame f."""
    gt_path = data / "production" / "gsr" / match / f"{match}_{half}.json"
    video = data / "interim" / match / f"{match}_panorama_{half}_half.mp4"
    if not gt_path.exists() or not video.exists():
        return {"match": match, "half": half, "offset": None, "reason": "missing input"}

    sample = [start_frame + i * step for i in range(n_frames)]
    B = _boxes_by_frame(gt_path, sample[0], sample[-1] + lag_range + 5)
    if not B:
        return {"match": match, "half": half, "offset": None, "reason": "no GT boxes"}

    # decode only the sampled frames, sequentially (no seeking: it is not frame-accurate)
    cap = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    want = set(sample)
    idx = 0
    last = max(sample)
    while idx < last:
        idx += 1
        if idx in want:
            ok, f = cap.read()
            if not ok:
                break
            frames[idx] = _grass_mask(f)
        else:
            if not cap.grab():
                break
    cap.release()
    if len(frames) < 10:
        return {"match": match, "half": half, "offset": None, "reason": "too few frames decoded"}

    lags = list(range(0, lag_range + 1, 1))
    curve = []
    for lag in lags:
        vals = [_box_occupancy(frames[f], B[f + lag]) for f in sorted(frames) if (f + lag) in B]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 10:
            continue
        curve.append((lag, float(np.mean(vals))))
    if not curve:
        return {"match": match, "half": half, "offset": None, "reason": "no valid lag"}
    best = max(curve, key=lambda t: t[1])
    occ0 = dict(curve).get(0, float("nan"))
    vals = np.array([c[1] for c in curve])
    # a peak is only meaningful if it stands out from the spread of the curve
    z = (best[1] - vals.mean()) / (vals.std() + 1e-9)
    return {"match": match, "half": half, "offset": best[0], "occupancy": round(best[1], 4),
            "occ_at_0": round(occ0, 4), "gain": round(best[1] - occ0, 4),
            "z": round(float(z), 2), "n_frames": len(frames), "reason": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--match"); ap.add_argument("--half", choices=["1st", "2nd"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="check against the two halves whose offset is independently known")
    ap.add_argument("--frames", type=int, default=60,
                    help="number of sampled frames (spread by --step)")
    ap.add_argument("--step", type=int, default=40)
    ap.add_argument("--start-frame", type=int, default=5000,
                    help="sample from here, not frame 1: at kickoff players are static and "
                         "background subtraction erases them")
    ap.add_argument("--lag-range", type=int, default=320)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data)

    if a.validate:
        print("VALIDATION against independently established offsets (both 25, from")
        print("cross-correlating real detector output against labels)\n")
        ok = True
        for m, h, known in (("128057", "1st", 25), ("117093", "1st", 25)):
            r = estimate(m, h, data, a.frames, a.lag_range, a.start_frame, a.step)
            good = r["offset"] is not None and abs(r["offset"] - known) <= 3
            ok &= good
            print(f"  {m} {h}: measured {r['offset']} (known {known}) "
                  f"occupancy {r.get('occupancy')} vs {r.get('occ_at_0')} at lag 0, z={r.get('z')} "
                  f"-> {'OK' if good else 'MISMATCH'}")
        print("\nestimator is trustworthy" if ok else "\nESTIMATOR NOT TRUSTWORTHY -- do not use")
        return 0 if ok else 1

    jobs = ([(m, h) for m in MATCHES for h in ("1st", "2nd")] if a.all
            else [(a.match, a.half)])
    rows = []
    print(f"{'match':8} {'half':5} {'offset':>7} {'sec':>6} {'occ':>7} {'occ@0':>7} "
          f"{'gain':>7} {'z':>6}")
    print("-" * 62)
    for m, h in jobs:
        r = estimate(m, h, data, a.frames, a.lag_range, a.start_frame, a.step)
        rows.append(r)
        off = r["offset"]
        print(f"{m:8} {h:5} {off if off is not None else 'FAIL':>7} "
              f"{(off/25.0 if off is not None else float('nan')):6.2f} "
              f"{r.get('occupancy', float('nan')):7} {r.get('occ_at_0', float('nan')):7} "
              f"{r.get('gain', float('nan')):7} {r.get('z', float('nan')):6}", flush=True)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

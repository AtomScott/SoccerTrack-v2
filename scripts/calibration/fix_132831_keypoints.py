"""Repair match 132831's camera calibration.

THE DEFECT
    Two entries in data/raw/132831/132831_keypoints.json hold each other's image
    coordinates:

        (88.5,13.84)  annotated [3356, 781]   belongs at [2668, 378]
        (105,54.16)   annotated [2640, 379]   belongs at [3346, 742]

    Both sit on real pitch features, so the annotation looks correct to the eye, and
    63 of the 65 points are fine. But the transposition is fatal to the least-squares
    fisheye fit:

        as shipped          cv2.fisheye.calibrate RMS = 1260.95 px
        with the two swapped                      RMS =   18.07 px

    At 1260.95 px the fit is meaningless and CALIB_CHECK_COND correctly refuses it, so
    src/calibration/generate_calibration_mappings.py raises. The unmerged
    `for_soccernet` branch removes CALIB_CHECK_COND, which converts that correct refusal
    into a silently degenerate calibration -- applying the resulting remap to 132831's
    own footage yields a radial smear with no recognisable pitch. Those corrupt maps were
    shipped into data/raw/132831/, data/interim/calibrated_keypoints/132831/ and
    data/production/raw/132831/, and 132831's GSR labels were generated through them.
    132831 is in the test split, so this contaminates evaluation.

    Evidence that the two points are the whole story: fitting 132831's 42 touchline
    points alone gives RMS 13.22 (healthy), fitting its 23 box/circle points alone gives
    1437.84, and projecting all 65 through the touchline-only fit leaves 21 points at
    13-44 px and exactly these two at ~795 px. Across all ten matches, 132831 is the only
    one with any gross outlier.

A SECOND, INDEPENDENT ISSUE (not fixed here)
    generate_calibration_mappings.py builds the undistortion map with balance=1 and an
    output canvas equal to the input size. For 132831's much wider field of view that
    combination puts the whole pitch into 22% of the frame width (healthy matches: 71%),
    leaving most of the canvas mapping outside the source image. The map is not folded --
    it is monotonic -- the pitch is simply tiny. Containing the pitch needs the canvas
    sized deliberately rather than inherited from the input; see --canvas-width below,
    and scripts/calibration/test_calibration_strategies.sh for the measurements.

USAGE
    # write corrected artifacts to a scratch dir and render a proof image (default)
    python scripts/calibration/fix_132831_keypoints.py

    # also overwrite the canonical location under $DATA (asks first)
    python scripts/calibration/fix_132831_keypoints.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# The two transposed keys. Their image coordinates are swapped with each other.
SWAP = ("(88.5,13.84)", "(105,54.16)")
MATCH = "132831"

FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_FIX_SKEW
    + cv2.fisheye.CALIB_CHECK_COND  # kept on: it is what caught this defect
    + cv2.fisheye.CALIB_FIX_K3
    + cv2.fisheye.CALIB_FIX_K4
)
CRITERIA = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)


def load(path: Path):
    with open(path) as f:
        d = json.load(f)
    keys = list(d)
    pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
    image = np.array(list(d.values()), float)
    return d, keys, pitch, image


def fit(pitch, image, size):
    objp = np.concatenate([pitch, np.zeros((len(pitch), 1))], 1).astype(np.float32).reshape(-1, 1, 3)
    imgp = image.astype(np.float32).reshape(-1, 1, 2)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    return cv2.fisheye.calibrate([objp], [imgp], size, K, D, None, None, flags=FLAGS, criteria=CRITERIA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out", default="outputs/calibration_test/fix_132831")
    ap.add_argument("--frame", type=int, default=30000, help="frame index for the proof render")
    ap.add_argument("--canvas-width", type=int, default=4096,
                    help="width of the undistorted canvas; the pitch is scaled to 94%% of it")
    ap.add_argument("--vertical-margin", type=float, default=3.0,
                    help="canvas height as a multiple of the undistorted pitch height")
    ap.add_argument("--apply", action="store_true",
                    help="also overwrite the canonical files under --data (prompts first)")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kp_path = data / "raw" / MATCH / f"{MATCH}_keypoints.json"

    d, keys, pitch, image = load(kp_path)
    for k in SWAP:
        if k not in d:
            print(f"FATAL: expected key {k} not present in {kp_path}")
            return 1

    video = data / "interim" / MATCH / f"{MATCH}_panorama_1st_half.mp4"
    if not video.exists():
        video = data / "raw" / MATCH / f"{MATCH}_panorama.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"FATAL: could not read frame {args.frame} from {video}")
        return 1
    h, w = frame.shape[:2]

    before, *_ = _try(pitch, image, (w, h))
    i, j = keys.index(SWAP[0]), keys.index(SWAP[1])
    fixed = image.copy()
    fixed[[i, j]] = fixed[[j, i]]
    after, K, D, rvecs, tvecs = fit(pitch, fixed, (w, h))

    print(f"{MATCH}: {w}x{h}, frame {args.frame}")
    print(f"  RMS as shipped : {before}")
    print(f"  RMS corrected  : {after:.2f}   (CALIB_CHECK_COND on, i.e. main's flags)")
    print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} k1={D.ravel()[0]:.4f}")

    corrected = dict(d)
    corrected[SWAP[0]] = list(d[SWAP[1]])
    corrected[SWAP[1]] = list(d[SWAP[0]])
    kp_out = out / f"{MATCH}_keypoints.json"
    with open(kp_out, "w") as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False)
    print(f"  wrote corrected keypoints -> {kp_out}")

    # Canonical artifacts, matching generate_calibration_mappings.py's output set but
    # with the canvas sized to contain the pitch rather than inherited from the input.
    nk0 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
    imgp = fixed.astype(np.float32).reshape(-1, 1, 2)
    u0 = cv2.fisheye.undistortPoints(imgp, K, D, P=nk0).reshape(-1, 2)
    bx0, by0, bx1, by1 = u0[:, 0].min(), u0[:, 1].min(), u0[:, 0].max(), u0[:, 1].max()
    pw, ph = bx1 - bx0, by1 - by0
    s = (args.canvas_width * 0.94) / pw
    cw = args.canvas_width
    ch = int(round(ph * s * args.vertical_margin))
    nk = nk0.copy()
    nk[0, 0] *= s
    nk[1, 1] *= s
    nk[0, 2] = cw / 2 - s * (((bx0 + bx1) / 2) - nk0[0, 2])
    nk[1, 2] = ch / 2 - s * (((by0 + by1) / 2) - nk0[1, 2])
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, (cw, ch), cv2.CV_16SC2)

    np.save(out / f"{MATCH}_mapx.npy", mapx)
    np.save(out / f"{MATCH}_mapy.npy", mapy)
    np.savez(out / f"{MATCH}_camera_intrinsics.npz", K=K, D=D, Knew=nk,
             rvecs=rvecs, tvecs=tvecs, rms=after)
    und = cv2.fisheye.undistortPoints(imgp, K, D, P=nk).reshape(-1, 2)
    with open(out / f"{MATCH}_calibrated_keypoints.json", "w") as f:
        json.dump({k: [float(a), float(b)] for k, (a, b) in zip(keys, und)}, f, indent=2)

    inside = float(((und[:, 0] >= 0) & (und[:, 0] < cw) & (und[:, 1] >= 0) & (und[:, 1] < ch)).mean())
    rect = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    proof = out / f"{MATCH}_proof.jpg"
    hh = 560
    cv2.imwrite(str(proof), cv2.resize(rect, (int(rect.shape[1] * hh / rect.shape[0]), hh)),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  canvas {cw}x{ch}, {inside:.0%} of keypoints inside")
    print(f"  wrote proof render -> {proof}")

    if args.apply:
        print("\n--apply given. This OVERWRITES shared dataset files:")
        targets = [
            data / "raw" / MATCH / f"{MATCH}_keypoints.json",
            data / "raw" / MATCH / f"{MATCH}_mapx.npy",
            data / "raw" / MATCH / f"{MATCH}_mapy.npy",
            data / "raw" / MATCH / f"{MATCH}_camera_intrinsics.npz",
        ]
        for t in targets:
            print(f"    {t}")
        print("  Each is backed up alongside with a .prefix suffix first.")
        if input("  type 'yes' to proceed: ").strip() != "yes":
            print("  aborted; nothing under --data was touched.")
            return 0
        for t in targets:
            if t.exists():
                shutil.copy2(t, t.with_suffix(t.suffix + ".broken-backup"))
        shutil.copy2(kp_out, targets[0])
        shutil.copy2(out / f"{MATCH}_mapx.npy", targets[1])
        shutil.copy2(out / f"{MATCH}_mapy.npy", targets[2])
        shutil.copy2(out / f"{MATCH}_camera_intrinsics.npz", targets[3])
        print("  applied. NOTE: 132831's GSR labels were derived from the broken "
              "calibration and must be regenerated separately.")
    return 0


def _try(pitch, image, size):
    """Fit and report, tolerating the expected CALIB_CHECK_COND failure."""
    try:
        return fit(pitch, image, size)
    except cv2.error:
        return ("FAILED (CALIB_CHECK_COND rejected it, as it should)", None, None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())

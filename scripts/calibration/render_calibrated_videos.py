"""Render calibrated (undistorted) per-half panorama videos for every match.

WHAT IT PRODUCES
    $DATA/interim/<match>/<match>_calibrated_panorama_{1st,2nd}_half.mp4

    That is the path baselines/gsr/config.yaml consumes, and the reason it insists on the
    calibrated variant is that the raw panorama is fisheye-distorted, so projecting through
    the homography on raw frames mis-locates everything.

HOW IT DIFFERS FROM scripts/calibrate_camera.sh
    * Calibration comes from each match's pitch keypoints via
      calibrate_all_from_keypoints.py's method, honouring data_corrections/ -- so 132831's
      two transposed labels are fixed rather than producing a degenerate map.
    * CALIB_CHECK_COND stays on, so a bad fit refuses to render instead of emitting a smear.
    * The output canvas is sized to contain the pitch rather than inherited from the input
      frame. See docs/calibration-findings.md, defect 2.

SAFETY
    * Existing outputs are skipped unless --force, so the run is resumable after an
      interruption.
    * If an output already exists and --force is given, the old file is renamed to
      <name>.pre-recalibration-backup rather than destroyed.
    * Writes only to <out-root>/<match>/. Reads everything else.

USAGE
    python scripts/calibration/render_calibrated_videos.py                  # all, resumable
    python scripts/calibration/render_calibrated_videos.py --matches 132831
    python scripts/calibration/render_calibrated_videos.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
HALVES = ["1st", "2nd"]
KEYPOINT_SWAPS = {"132831": [("(88.5,13.84)", "(105,54.16)")]}

FLAGS = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
         + cv2.fisheye.CALIB_FIX_SKEW
         + cv2.fisheye.CALIB_CHECK_COND
         + cv2.fisheye.CALIB_FIX_K3
         + cv2.fisheye.CALIB_FIX_K4)
CRITERIA = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)


def load_keypoints(data: Path, match: str, corrections: Path):
    corrected = corrections / f"{match}_keypoints.json"
    if corrected.exists():
        d = json.load(open(corrected))
        keys = list(d)
        pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
        return pitch, np.array(list(d.values()), float), f"corrections/{corrected.name}"
    d = json.load(open(data / "raw" / match / f"{match}_keypoints.json"))
    keys = list(d)
    pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
    image = np.array(list(d.values()), float)
    src = "dataset"
    for a, b in KEYPOINT_SWAPS.get(match, []):
        if a in d and b in d:
            i, j = keys.index(a), keys.index(b)
            image[[i, j]] = image[[j, i]]
            src = "dataset + in-code swap"
    return pitch, image, src


def build_maps(pitch, image, w, h, canvas_width, vmargin):
    objp = np.concatenate([pitch, np.zeros((len(pitch), 1))], 1).astype(np.float32).reshape(-1, 1, 3)
    imgp = image.astype(np.float32).reshape(-1, 1, 2)
    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    rms, K, D, _, _ = cv2.fisheye.calibrate([objp], [imgp], (w, h), K, D, None, None,
                                            flags=FLAGS, criteria=CRITERIA)
    nk0 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
    u0 = cv2.fisheye.undistortPoints(imgp, K, D, P=nk0).reshape(-1, 2)
    pw, ph = u0[:, 0].ptp(), u0[:, 1].ptp()
    s = (canvas_width * 0.94) / pw
    cw = canvas_width + (canvas_width % 2)
    ch = int(round(ph * s * vmargin)); ch += ch % 2          # even dims required by the encoder
    nk = nk0.copy(); nk[0, 0] *= s; nk[1, 1] *= s
    nk[0, 2] = cw / 2 - s * ((u0[:, 0].min() + u0[:, 0].max()) / 2 - nk0[0, 2])
    nk[1, 2] = ch / 2 - s * ((u0[:, 1].min() + u0[:, 1].max()) / 2 - nk0[1, 2])
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, (cw, ch), cv2.CV_16SC2)
    return rms, mapx, mapy, cw, ch


def render(src_video: Path, dst: Path, mapx, mapy, cw, ch, fps, encoder, cq, log):
    tmp = dst.with_suffix(".partial.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}", "-r", f"{fps}",
           "-i", "pipe:0"]
    if encoder == "h264_nvenc":
        # -b:v 0 is REQUIRED. Without it NVENC's vbr mode ignores -cq and targets a huge
        # default bitrate: measured 9.08 GB per half versus 1.74 GB with it, for identical
        # -cq 23. The 1.74 GB matches what the previous pipeline produced (1.64 GB), so
        # this is a size fix, not a quality change.
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(cq),
                "-b:v", "0", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cq), "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", str(tmp)]

    cap = cv2.VideoCapture(str(src_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time(); n = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            proc.stdin.write(cv2.remap(fr, mapx, mapy, interpolation=cv2.INTER_LINEAR).tobytes())
            n += 1
            if n % 5000 == 0:
                el = time.time() - t0
                eta = (total - n) / (n / el) / 60 if n else 0
                print(f"      {n}/{total} frames  {n/el:6.1f} fps  eta {eta:5.1f} min", flush=True)
                log.write(f"      {n}/{total} {n/el:.1f} fps\n"); log.flush()
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait()
    if rc != 0 or not tmp.exists():
        if tmp.exists():
            tmp.unlink()
        return None, n, time.time() - t0
    tmp.rename(dst)
    return dst, n, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out-root", default=None, help="default: <data>/interim")
    ap.add_argument("--corrections", default="data_corrections")
    ap.add_argument("--matches", nargs="*", default=MATCHES)
    ap.add_argument("--canvas-width", type=int, default=4096)
    ap.add_argument("--vertical-margin", type=float, default=3.0)
    ap.add_argument("--encoder", default="h264_nvenc", choices=["h264_nvenc", "libx264"])
    ap.add_argument("--cq", type=int, default=23)
    ap.add_argument("--force", action="store_true", help="re-render, backing up any existing output")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default="outputs/render_calibrated_videos.log")
    args = ap.parse_args()

    data = Path(args.data)
    out_root = Path(args.out_root) if args.out_root else data / "interim"
    corrections = Path(args.corrections)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a")
    log.write(f"\n=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    todo, skipped = [], []
    for m in args.matches:
        for half in HALVES:
            src = data / "interim" / m / f"{m}_panorama_{half}_half.mp4"
            dst = out_root / m / f"{m}_calibrated_panorama_{half}_half.mp4"
            if not src.exists():
                skipped.append((m, half, "source missing"))
            elif dst.exists() and not args.force:
                skipped.append((m, half, "already rendered"))
            else:
                todo.append((m, half, src, dst))

    print(f"to render: {len(todo)}   skipping: {len(skipped)}")
    for m, half, why in skipped:
        print(f"  skip {m} {half}: {why}")
    if args.dry_run:
        for m, half, src, dst in todo:
            print(f"  would render {src.name} -> {dst}")
        return 0

    done, failed = [], []
    maps_cache = {}
    for idx, (m, half, src, dst) in enumerate(todo, 1):
        print(f"\n[{idx}/{len(todo)}] {m} {half} half", flush=True)
        cap = cv2.VideoCapture(str(src))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        key = (m, w, h)
        if key not in maps_cache:
            pitch, image, ksrc = load_keypoints(data, m, corrections)
            try:
                maps_cache[key] = build_maps(pitch, image, w, h, args.canvas_width, args.vertical_margin)
            except cv2.error as e:
                msg = str(e).splitlines()[-1][:80]
                print(f"    CALIBRATION REFUSED: {msg}")
                log.write(f"{m} {half}: CALIBRATION REFUSED {msg}\n")
                failed.append((m, half, "calibration refused")); continue
            print(f"    keypoints: {ksrc}   rms={maps_cache[key][0]:.2f}")
        rms, mapx, mapy, cw, ch = maps_cache[key]
        print(f"    {w}x{h} -> {cw}x{ch} @ {fps:.0f}fps  encoder={args.encoder}", flush=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and args.force:
            bk = dst.with_name(dst.name + ".pre-recalibration-backup")
            if not bk.exists():
                shutil.move(str(dst), str(bk))
                print(f"    backed up existing -> {bk.name}")
        out, n, dt = render(src, dst, mapx, mapy, cw, ch, fps, args.encoder, args.cq, log)
        if out is None:
            print(f"    FAILED after {n} frames")
            log.write(f"{m} {half}: FAILED after {n} frames\n"); failed.append((m, half, "encode failed"))
        else:
            size = out.stat().st_size / 1e9
            print(f"    done: {n} frames in {dt/60:.1f} min ({n/dt:.1f} fps), {size:.2f} GB")
            log.write(f"{m} {half}: OK {n} frames {dt/60:.1f} min {size:.2f} GB -> {out}\n")
            done.append((m, half, out))
        log.flush()

    print(f"\n===== rendered {len(done)}, failed {len(failed)}, skipped {len(skipped)} =====")
    for m, half, why in failed:
        print(f"  FAILED {m} {half}: {why}")
    log.write(f"=== finished: {len(done)} ok, {len(failed)} failed ===\n")
    log.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""GSR baseline pipeline for SoccerTrack v2.

Implements the five documented stages of the Game State Reconstruction (GSR)
baseline as *importable, independently-callable functions* plus a thin argparse
``main`` that reads ``config.yaml``:

    1. ``build_yolo_dataset``  — build a YOLO detection dataset from the MOT
       annotations of the train/val matches, via
       ``src.data_utils.create_yolo_dataset.create_yolo_dataset``.
    2. ``train_detector``      — fine-tune YOLOv8-s with ultralytics.
    3. ``run_tracker``         — run ByteTrack over each test half (ultralytics
       ``model.track(..., tracker="bytetrack.yaml")``) → MOT-style tracklets.
    4. ``tag_tracklets``       — tag each tracklet with jersey / role / team via a
       *pluggable* Re-ID step (CLIP-ReID per the paper). The default
       implementation is a clearly-marked stub returning "unknown" identity.
    5. ``project_to_pitch`` + ``write_gsr_json`` — project image-plane feet to
       pitch metres with the per-match homography and emit GSR-format JSON,
       one file per half per match, under
       ``outputs/gsr_baseline/preds/<match>/<match>_{1st,2nd}.json`` — exactly
       the layout ``src.evaluation.gs_hota`` / the format spec expect.

Design constraints (so this stays CPU-importable and unit-testable):

* Heavy / optional deps (``torch``, ``ultralytics``, ``cv2``) are imported
  *inside* the functions that need them, so ``python -m baselines.gsr.train
  --help`` works in a bare environment and individual pure-Python stages
  (projection, JSON emission, the Re-ID stub) can be unit-tested on tiny
  synthetic inputs with only numpy installed.
* Every stage is a free function with explicit args; ``main`` only wires
  config → stage calls.

Homography convention
---------------------
The pitch projection uses a **3x3 pitch->image homography** stored as a numpy
``.npy`` at ``<homography.root>/<match>/<match>_homography.npy`` (default root
``data/interim/homography``). This is the *same* convention used by the rest of
the repo — see ``src/coordinate_conversion/convert_pitch_plane_to_image_plane.py``
and ``src/data_association/create_ground_truth.py`` (both ``np.load`` a 3x3 ``H``
and use ``cv2.perspectiveTransform`` with ``H`` / ``inv(H)``).

NOTE: the original starter ``config.yaml`` pointed at ``raw/<match>/<match>_mapx.npy``
+ ``mapy.npy``. Those files are **fisheye undistortion remap tables** produced by
``src/calibration/generate_calibration_mappings.py`` (dense ``H x W x 2`` maps for
``cv2.remap``), *not* a planar homography — they cannot be fed to
``cv2.perspectiveTransform``. The config has been corrected to the homography
convention above; see ``config.yaml`` for the reconciled key
(``homography.root`` / ``homography.filename_template``).

Run:
    python -m baselines.gsr.train --config baselines/gsr/config.yaml
    python -m baselines.gsr.train --config baselines/gsr/config.yaml --stages dataset detect track tag project
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import yaml

# --------------------------------------------------------------------------- #
# Lightweight data structures (no heavy deps)                                  #
# --------------------------------------------------------------------------- #

# A single tracker observation in image pixels. Mirrors the MOT-ish layout used
# elsewhere in the repo (frame, id, bb_left/top/width/height, conf, class_name).
@dataclass(frozen=True)
class Detection:
    frame: int          # 0-indexed frame within the half
    track_id: int       # tracker-assigned id (per half)
    bb_left: float      # pixels, top-left x on the (panoramic) image plane
    bb_top: float       # pixels, top-left y
    bb_width: float     # pixels
    bb_height: float    # pixels
    conf: float = 1.0
    class_name: str = "person"


@dataclass(frozen=True)
class Identity:
    """Re-ID output for one tracklet (constant within a tracklet)."""

    role: str = "player"                 # player|goalkeeper|referee|other
    jersey_number: Optional[int] = None  # 0..99 or None
    team_side: Optional[str] = None      # left|right or None


# --------------------------------------------------------------------------- #
# Config helpers                                                               #
# --------------------------------------------------------------------------- #

_HALF_SUFFIX = {1: "1st", 2: "2nd"}
_PERIOD_FOR_HALF = {1: "1st_half", 2: "2nd_half"}


def load_config(config_path: str | Path) -> dict:
    """Read the YAML config into a plain dict."""
    return yaml.safe_load(Path(config_path).read_text())


def _data_root(cfg: dict) -> Path:
    return Path(cfg["data"]["root"])


def homography_path(cfg: dict, match_id: str) -> Path:
    """Resolve the per-match homography ``.npy`` path from config.

    Default convention (repo-wide): ``data/interim/homography/<match>/<match>_homography.npy``.
    Overridable via ``cfg['homography']['root']`` and
    ``cfg['homography']['filename_template']`` (``{match}`` placeholder).
    """
    h = cfg.get("homography", {})
    root = Path(h.get("root", str(_data_root(cfg) / "interim" / "homography")))
    template = h.get("filename_template", "{match}_homography.npy")
    return root / str(match_id) / template.format(match=match_id)


def video_path(cfg: dict, match_id: str, half: int) -> Path:
    """Resolve the (calibrated) panoramic video for a half.

    Default: ``<data.root>/interim/calibrated_videos/<match>/<match>_panorama_<half>.mp4``.
    Overridable via ``cfg['video']['root']`` / ``cfg['video']['filename_template']``
    (``{match}`` and ``{half}`` placeholders; ``{half}`` is ``1st_half``/``2nd_half``).
    """
    v = cfg.get("video", {})
    root = Path(v.get("root", str(_data_root(cfg) / "interim" / "calibrated_videos")))
    template = v.get("filename_template", "{match}_panorama_{half}.mp4")
    return root / str(match_id) / template.format(match=match_id, half=_PERIOD_FOR_HALF[half])


def mot_path(cfg: dict, match_id: str, half: int) -> Path:
    """Resolve the MOT ground-truth annotation file for a half (for dataset build)."""
    m = cfg.get("mot", {})
    root = Path(m.get("root", str(_data_root(cfg) / "mot")))
    template = m.get("filename_template", "{match}_{half}.txt")
    return root / str(match_id) / template.format(match=match_id, half=_PERIOD_FOR_HALF[half])


# --------------------------------------------------------------------------- #
# Stage 1 — build YOLO detection dataset from MOT annotations                  #
# --------------------------------------------------------------------------- #

def build_yolo_dataset(cfg: dict, out_dir: str | Path | None = None) -> list[Path]:
    """Build one YOLO dataset per (train/val) match-half from MOT annotations.

    Wraps ``src.data_utils.create_yolo_dataset.create_yolo_dataset``. We call it
    once per (match, half) because that helper takes a single ``video_path`` +
    ``mot_path`` pair and writes its own ``data.yaml``; the resulting per-half
    dataset directories can then be merged / listed for training.

    Returns the list of created dataset directories. Requires the data snapshot
    (videos + MOT files) and the helper's heavy deps (cv2, deffcode) to be
    present — so this is *not* exercised in the CPU unit tests, but it is fully
    wired and import-safe (the heavy import is deferred to call time).
    """
    # Deferred import: create_yolo_dataset pulls in cv2/pandas/sklearn/deffcode.
    from src.data_utils.create_yolo_dataset import create_yolo_dataset

    out_dir = Path(out_dir) if out_dir is not None else Path(cfg["detector"].get(
        "dataset_dir", "outputs/gsr_baseline/yolo_dataset"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["detector"].get("dataset", {})
    frame_interval = int(ds_cfg.get("frame_interval", 5))
    splits = ds_cfg.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    frame_w = ds_cfg.get("frame_width")
    frame_h = ds_cfg.get("frame_height")

    created: list[Path] = []
    matches = list(cfg["data"]["train_matches"]) + list(cfg["data"]["val_matches"])
    for match_id in matches:
        for half in (1, 2):
            vpath = video_path(cfg, str(match_id), half)
            mpath = mot_path(cfg, str(match_id), half)
            if not vpath.exists() or not mpath.exists():
                # Be explicit rather than silently skipping — the caller chose
                # these matches.
                raise FileNotFoundError(
                    f"Missing inputs for match {match_id} half {half}: "
                    f"video={vpath} (exists={vpath.exists()}), "
                    f"mot={mpath} (exists={mpath.exists()})"
                )
            half_out = out_dir / f"{match_id}_{_HALF_SUFFIX[half]}"
            create_yolo_dataset(
                video_path=str(vpath),
                mot_path=str(mpath),
                output_dir=str(half_out),
                frame_interval=frame_interval,
                train_split=float(splits["train"]),
                val_split=float(splits["val"]),
                test_split=float(splits["test"]),
                frame_width=frame_w,
                frame_height=frame_h,
                overwrite=True,
            )
            created.append(half_out)
    return created


# --------------------------------------------------------------------------- #
# Stage 2 — fine-tune the detector                                            #
# --------------------------------------------------------------------------- #

def train_detector(cfg: dict, data_yaml: str | Path) -> Path:
    """Fine-tune YOLOv8-s on ``data_yaml`` via ultralytics. Returns weights path.

    Deferred ultralytics import keeps ``--help`` cheap. ``data_yaml`` is a path
    to an ultralytics dataset spec (e.g. produced by ``build_yolo_dataset``).
    """
    from ultralytics import YOLO  # deferred heavy import

    det = cfg["detector"]
    base = det.get("base_weights", "yolov8s.pt")
    model = YOLO(base)
    project = Path(det.get("project", "outputs/gsr_baseline/detector"))
    name = det.get("run_name", "yolov8s_gsr")
    results = model.train(
        data=str(data_yaml),
        epochs=int(det.get("epochs", 50)),
        imgsz=int(det.get("imgsz", 1280)),
        batch=int(det.get("batch", 8)),
        project=str(project),
        name=name,
        exist_ok=True,
    )
    # Ultralytics writes best.pt under <project>/<name>/weights/best.pt.
    best = project / name / "weights" / "best.pt"
    if not best.exists():
        # Fall back to whatever ultralytics reports.
        save_dir = getattr(results, "save_dir", None)
        if save_dir is not None:
            cand = Path(save_dir) / "weights" / "best.pt"
            if cand.exists():
                best = cand
    return best


# --------------------------------------------------------------------------- #
# Stage 3 — track each test half with ByteTrack                               #
# --------------------------------------------------------------------------- #

def run_tracker(
    cfg: dict,
    weights_path: str | Path,
    video: str | Path,
) -> list[Detection]:
    """Run ByteTrack (via ultralytics ``model.track``) over one half video.

    Returns a flat list of :class:`Detection` (per-frame, per-track boxes in
    image pixels). Deferred ultralytics import. Only "person"-class boxes that
    received a track id are kept (the ball is class 1 in the dataset and is not a
    GSR entity).
    """
    from ultralytics import YOLO  # deferred heavy import

    tracker = cfg.get("tracker", {})
    tracker_name = tracker.get("name", "bytetrack")
    # ultralytics ships bytetrack.yaml / botsort.yaml; allow an explicit path.
    tracker_yaml = tracker.get("config", f"{tracker_name}.yaml")
    det = cfg["detector"]

    model = YOLO(str(weights_path))
    out: list[Detection] = []
    results = model.track(
        source=str(video),
        tracker=tracker_yaml,
        conf=float(det.get("conf_thresh", 0.35)),
        iou=float(det.get("iou_thresh", 0.55)),
        imgsz=int(det.get("imgsz", 1280)),
        stream=True,
        verbose=False,
        persist=True,
    )
    for frame_idx, res in enumerate(results):
        if res.boxes is None:
            continue
        for box in res.boxes:
            if box.id is None:
                continue
            cls_name = res.names[int(box.cls[0])]
            if cls_name != "person":
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(
                Detection(
                    frame=frame_idx,
                    track_id=int(box.id[0]),
                    bb_left=x1,
                    bb_top=y1,
                    bb_width=x2 - x1,
                    bb_height=y2 - y1,
                    conf=float(box.conf[0]),
                    class_name=cls_name,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Stage 4 — tag tracklets with jersey / role via a pluggable Re-ID step        #
# --------------------------------------------------------------------------- #

def stub_reid(track_ids: Iterable[int]) -> dict[int, Identity]:
    """Default Re-ID: returns "unknown" identities for every tracklet.

    *** STUB — NOT A REAL RE-ID MODEL. ***

    The paper uses CLIP-ReID for jersey-number / role / team tagging. Wiring real
    CLIP-ReID weights is non-trivial (weights aren't shipped here), so this stub
    keeps the pipeline runnable end-to-end while producing *honest* unknown
    identities: ``role="player"`` (the dominant class; a defensible prior),
    ``jersey_number=None`` (unobserved), ``team_side=None`` (unknown).

    Per the GSR metric (GS-HOTA), an unknown/incorrect jersey+team+role is scored
    as an identification mismatch, so a real submission MUST replace this with
    :func:`tag_tracklets`'s ``reid_fn`` argument pointing at a trained model.

    Returns ``{track_id: Identity}``.
    """
    return {tid: Identity(role="player", jersey_number=None, team_side=None)
            for tid in set(track_ids)}


def tag_tracklets(
    detections: Sequence[Detection],
    reid_fn: Callable[[Iterable[int]], dict[int, Identity]] = stub_reid,
    crops: Optional[object] = None,
) -> dict[int, Identity]:
    """Assign one :class:`Identity` per track id.

    ``reid_fn`` is the pluggable Re-ID function. The default (:func:`stub_reid`)
    returns unknown identities and needs no model or image crops. A real
    implementation would consume per-track image crops (``crops``) and return a
    confident ``Identity`` per track; pass it as ``reid_fn``.

    This function is pure given ``reid_fn`` and so is directly unit-testable.
    """
    track_ids = [d.track_id for d in detections]
    identities = reid_fn(track_ids)
    # Guarantee every observed track gets *some* identity (defensive).
    for tid in set(track_ids):
        identities.setdefault(tid, Identity())
    return identities


# --------------------------------------------------------------------------- #
# Stage 5 — project to pitch metres + emit GSR-format JSON                     #
# --------------------------------------------------------------------------- #

def _feet_point(d: Detection) -> tuple[float, float]:
    """Bottom-centre of the bbox in image pixels (approx. the player's feet)."""
    return (d.bb_left + d.bb_width / 2.0, d.bb_top + d.bb_height)


def project_points_to_pitch(
    points_xy,
    homography,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    center_origin: bool = True,
):
    """Project image-plane points to pitch metres using a pitch->image homography.

    Mirrors ``src/data_association/create_ground_truth.py``: the stored ``H`` is a
    **pitch->image** homography, so image->pitch uses ``inv(H)`` via
    ``cv2.perspectiveTransform``. The repo's homography maps *metric pitch
    coordinates in pixels-of-metres scale* — i.e. ``cv2.perspectiveTransform(img,
    inv(H))`` yields coordinates in the range ``[0, pitch_length] x [0,
    pitch_width]`` with origin at a pitch *corner*.

    The SoccerTrack GSR format (docs/format-gsr.md) uses a **centre-origin**
    metric frame: ``x in [-pitch_length/2, +pitch_length/2]``,
    ``y in [-pitch_width/2, +pitch_width/2]``. When ``center_origin=True`` we
    shift the corner-origin result to centre-origin and flip ``y`` so it grows
    "up" as the spec requires (image/corner ``y`` grows down).

    Args:
        points_xy: ``(N, 2)`` array-like of image-plane pixel coordinates.
        homography: ``(3, 3)`` pitch->image homography (numpy array).
        pitch_length / pitch_width: pitch dimensions in metres.
        center_origin: if True, return centre-origin metric coords per the spec.

    Returns:
        ``(N, 2)`` numpy array of pitch coordinates in metres.

    Note:
        This is the one stage that needs ``cv2`` + ``numpy``. It is import-safe
        (deferred) and unit-tested with an identity-like homography below.
    """
    import numpy as np
    import cv2

    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    H = np.asarray(homography, dtype=np.float64)
    H_inv = np.linalg.inv(H)
    proj = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2).astype(np.float64)
    if center_origin:
        # corner-origin (0..L, 0..W, y-down) -> centre-origin (y-up)
        proj[:, 0] = proj[:, 0] - pitch_length / 2.0
        proj[:, 1] = (pitch_width / 2.0) - proj[:, 1]
    return proj


def build_gsr_records(
    detections: Sequence[Detection],
    identities: dict[int, Identity],
    homography,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> list[dict]:
    """Turn tracker detections + identities + homography into GSR JSON records.

    Each record follows ``docs/format-gsr.md``: ``image_id``, ``track_id``,
    ``role``, ``jersey_number``, ``team_side``, ``x``, ``y``, ``bbox_image``,
    ``bbox_pitch``. Positions are the projected feet point in centre-origin
    pitch metres.

    Pure given a homography array (no I/O); unit-tested below.
    """
    import numpy as np

    if not detections:
        return []

    feet = [_feet_point(d) for d in detections]
    pitch = project_points_to_pitch(
        feet, homography, pitch_length=pitch_length, pitch_width=pitch_width
    )

    records: list[dict] = []
    for d, (px, py) in zip(detections, pitch):
        ident = identities.get(d.track_id, Identity())
        # Project the bbox base corners to estimate a pitch-plane footprint.
        bl = (d.bb_left, d.bb_top + d.bb_height)
        br = (d.bb_left + d.bb_width, d.bb_top + d.bb_height)
        corners = project_points_to_pitch(
            [bl, br], homography, pitch_length=pitch_length, pitch_width=pitch_width
        )
        foot_w = float(abs(corners[1][0] - corners[0][0]))
        records.append(
            {
                "image_id": int(d.frame),
                "track_id": int(d.track_id),
                "role": ident.role,
                "jersey_number": ident.jersey_number,
                "team_side": ident.team_side,
                "x": float(px),
                "y": float(py),
                "bbox_image": [
                    int(round(d.bb_left)),
                    int(round(d.bb_top)),
                    int(round(d.bb_width)),
                    int(round(d.bb_height)),
                ],
                "bbox_pitch": [float(px), float(py), foot_w, 0.0],
            }
        )
    records.sort(key=lambda r: (r["image_id"], r["track_id"]))
    return records


def write_gsr_json(records: list[dict], pred_root: str | Path, match_id: str, half: int) -> Path:
    """Write GSR records to ``<pred_root>/<match>/<match>_{1st,2nd}.json``.

    This is exactly the layout the format spec and ``src.evaluation.gs_hota``
    consume (one JSON array per half per match). Returns the written path.
    """
    pred_root = Path(pred_root)
    out_dir = pred_root / str(match_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{match_id}_{_HALF_SUFFIX[half]}.json"
    out_path.write_text(json.dumps(records, indent=2))
    return out_path


# --------------------------------------------------------------------------- #
# End-to-end per-half driver (used by main; needs the heavy deps + data)       #
# --------------------------------------------------------------------------- #

def predict_half(
    cfg: dict,
    weights_path: str | Path,
    match_id: str,
    half: int,
    reid_fn: Callable[[Iterable[int]], dict[int, Identity]] = stub_reid,
) -> Path:
    """Stages 3-5 for one test half → written GSR JSON path."""
    import numpy as np

    vpath = video_path(cfg, match_id, half)
    if not vpath.exists():
        raise FileNotFoundError(f"Test video not found: {vpath}")
    hpath = homography_path(cfg, match_id)
    if not hpath.exists():
        raise FileNotFoundError(
            f"Homography not found: {hpath}. See module docstring / config.yaml "
            f"for the expected convention."
        )
    homography = np.load(hpath)

    pitch = cfg.get("pitch", {})
    pl = float(pitch.get("length", 105.0))
    pw = float(pitch.get("width", 68.0))

    detections = run_tracker(cfg, weights_path, vpath)
    identities = tag_tracklets(detections, reid_fn=reid_fn)
    records = build_gsr_records(
        detections, identities, homography, pitch_length=pl, pitch_width=pw
    )
    pred_root = Path(cfg["eval"].get("pred_root", "outputs/gsr_baseline/preds"))
    return write_gsr_json(records, pred_root, match_id, half)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

_ALL_STAGES = ("dataset", "detect", "track", "tag", "project")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="GSR baseline pipeline (SoccerTrack v2).")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        choices=_ALL_STAGES,
        help=(
            "Subset of stages to run, in order: "
            f"{', '.join(_ALL_STAGES)}. Default: all. "
            "('track' and 'project' are merged in the per-half driver.)"
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Detector weights for track/project stages (overrides cfg.detector.weights).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stages = tuple(args.stages) if args.stages else _ALL_STAGES
    print(f"[gsr] config: {args.config}")
    print(f"[gsr] stages: {stages}")

    data_yaml: Optional[Path] = None
    weights: Optional[Path] = args.weights or (
        Path(cfg["detector"]["weights"]) if cfg["detector"].get("weights") else None
    )

    if "dataset" in stages:
        print("[gsr] stage 1/5: build YOLO dataset from MOT annotations")
        created = build_yolo_dataset(cfg)
        # Train on the first dataset's data.yaml by default; a real run would
        # merge per-half datasets (out of scope for the thin baseline).
        if created:
            data_yaml = created[0] / "data.yaml"
        print(f"[gsr]   built {len(created)} per-half datasets")

    if "detect" in stages:
        print("[gsr] stage 2/5: fine-tune detector")
        if data_yaml is None:
            ddir = Path(cfg["detector"].get("dataset_dir", "outputs/gsr_baseline/yolo_dataset"))
            data_yaml = next(ddir.glob("*/data.yaml"), None)
        if data_yaml is None or not Path(data_yaml).exists():
            raise SystemExit(
                "[gsr] no dataset data.yaml found; run the 'dataset' stage first "
                "or set detector.dataset_dir."
            )
        weights = train_detector(cfg, data_yaml)
        print(f"[gsr]   weights: {weights}")

    if "track" in stages or "project" in stages:
        if weights is None or not Path(weights).exists():
            raise SystemExit(
                "[gsr] detector weights required for track/project. Pass --weights "
                "or set detector.weights / run the 'detect' stage."
            )
        print("[gsr] stages 3-5/5: track + tag + project, per test half")
        for match_id in cfg["data"]["test_matches"]:
            for half in (1, 2):
                out = predict_half(cfg, weights, str(match_id), half)
                print(f"[gsr]   wrote {out}")

    print("[gsr] done.")


if __name__ == "__main__":
    main()

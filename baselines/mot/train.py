"""MOT baseline pipeline for SoccerTrack v2.

Multi-object tracking is the *tracking-only subset* of the GSR baseline: build a
YOLO detection dataset from the MOT annotations, fine-tune YOLOv8-s, run
ByteTrack on the held-out test halves, then write per-sequence predictions — but
**without** the GSR-specific Re-ID, pitch homography, or pitch projection. The
output is plain image-plane tracks, emitted in the MOTChallenge ``gt.txt`` column
layout that :mod:`src.evaluation.mot_hota` (and TrackEval's ``MotChallenge2DBox``)
parse.

Reuse, not duplication
----------------------
The detection/tracking stages are *imported* from :mod:`baselines.gsr.train`
rather than re-implemented here — they are identical work:

* :func:`baselines.gsr.train.build_yolo_dataset` — MOT annotations -> YOLO dataset
  (wraps ``src.data_utils.create_yolo_dataset.create_yolo_dataset``).
* :func:`baselines.gsr.train.train_detector`   — fine-tune YOLOv8-s (ultralytics).
* :func:`baselines.gsr.train.run_tracker`      — ByteTrack over one half video,
  returning a flat list of :class:`~baselines.gsr.train.Detection` (image-plane
  boxes, per-half 0-indexed frames). It keeps only ``"person"`` class tracks (the
  ball, class 1 in the dataset, is correctly dropped — MOT scores person tracks
  with ``class = 1``).
* config helpers :func:`~baselines.gsr.train.load_config`,
  :func:`~baselines.gsr.train.video_path`, :func:`~baselines.gsr.train.mot_path`.

The **only** logic original to this module is the MOTChallenge writer
(:func:`detections_to_mot_rows` / :func:`write_mot_predictions`) and the per-half
sequence-naming / driver glue. There is no Re-ID and no homography here, so this
module imports cleanly on a CPU-only box with no torch/ultralytics/cv2 — the heavy
deps stay deferred inside the reused GSR functions.

MOTChallenge format (see docs/format-mot.md)
--------------------------------------------
Each prediction row, comma-separated, no header::

    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>

* ``frame`` is **1-indexed** (MOTChallenge convention). The tracker's per-half
  ``Detection.frame`` is **0-indexed**, so we add 1 here — the single most common
  conversion bug (see docs/format-mot.md "Known edge cases").
* ``id`` is the tracker track id, persistent within the sequence.
* ``bb_left, bb_top, bb_width, bb_height`` are image-plane pixels (cols 3-6),
  exactly the tracker output (and GSR's ``bbox_image``).
* ``conf`` is the detection confidence (a float; TrackEval may threshold it).
* ``class`` is ``1`` ("pedestrian"/person) for every tracked entity — the dataset
  collapses player/goalkeeper/referee into one class; role is *not* recoverable
  from MOT (use GSR for role).
* ``visibility`` is ``1`` for predictions (unknown; ground truth carries the real
  ratio).

Sequence layout
---------------
SoccerTrack v2 keys GSR/BAS per half but MOT sequences per ``<match_id>``. Because
the released MOT ground truth is per-half (``data/mot/<match>/<half>/gt/gt.txt``,
see baselines/gsr/config.yaml), and TrackEval pairs a ``gt/gt.txt`` against a
``data.txt`` by *sequence name*, we score **each half as its own sequence** named
``<match>_<half>`` (e.g. ``117099_1st``) — exactly the convention docs/format-mot.md
recommends for separately-evaluated halves. Predictions are written in the native
MOTChallenge ``trackers/<tracker>/data/<seq>.txt`` layout that TrackEval expects::

    <pred_root>/<tracker>/data/<match>_<half>.txt

(e.g. ``<pred_root>/bytetrack/data/117099_1st.txt``). For this to be scoreable,
the ground truth must sit at ``<gt_root>/<match>_<half>/gt/gt.txt`` with a
``seqinfo.ini`` per sequence, and :mod:`src.evaluation.mot_hota` must set
``SKIP_SPLIT_FOL=True`` so neither side is nested under a ``MOT17-custom`` split
folder (it does). The companion ``baselines/mot/eval.py`` reads
``cfg.data.test_matches`` as the list of sequence names, so ``config.yaml`` lists
the per-half sequence names there (e.g. ``117099_1st``) and :func:`sequence_name`
keeps the train.py naming in lockstep.

Run::

    python -m baselines.mot.train --config baselines/mot/config.yaml
    python -m baselines.mot.train --config baselines/mot/config.yaml --stages dataset detect track
    python -m baselines.mot.train --config baselines/mot/config.yaml --stages track --weights path/to/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

# Reuse the GSR detection/tracking stages verbatim. These imports are
# CPU-safe: baselines.gsr.train defers torch/ultralytics/cv2 inside the
# functions that need them, so importing the module (and these names) does not
# pull in any GPU/heavy dependency. Verified: `python -m baselines.mot.train
# --help` works in a bare env.
from baselines.gsr.train import (
    Detection,
    build_yolo_dataset,
    load_config,
    mot_path,
    run_tracker,
    train_detector,
    video_path,
)

# Half index -> sequence-name suffix. Mirrors baselines.gsr.train._HALF_SUFFIX so
# the per-half sequence names match the per-half MOT ground-truth dirs.
_HALF_SUFFIX = {1: "1st", 2: "2nd"}

# MOTChallenge constants for predictions.
_MOT_CLASS_PERSON = 1   # docs/format-mot.md: all tracked entities are class 1.
_MOT_VISIBILITY = 1     # predictions report full visibility (unknown -> 1).


# --------------------------------------------------------------------------- #
# Sequence naming                                                             #
# --------------------------------------------------------------------------- #

def sequence_name(match_id: str | int, half: int) -> str:
    """Sequence name for one half, e.g. ``("117099", 1) -> "117099_1st"``.

    This is the key TrackEval uses to pair a prediction ``data.txt`` against a
    ground-truth ``gt/gt.txt`` (docs/format-mot.md). It must match the per-half
    MOT ground-truth directory layout and the ``--matches`` / ``test_matches``
    entries given to the evaluator.
    """
    return f"{match_id}_{_HALF_SUFFIX[half]}"


_SUFFIX_HALF = {v: k for k, v in _HALF_SUFFIX.items()}  # {"1st": 1, "2nd": 2}


def parse_test_sequence(seq: str | int) -> tuple[str, int]:
    """Split a configured test sequence into ``(match_id, half)``.

    The MOT evaluator (baselines/mot/eval.py) consumes ``cfg.data.test_matches``
    as TrackEval **sequence names**, so to keep train.py and eval.py in lockstep
    we list per-half sequence names there (e.g. ``117099_1st``) and recover the
    match id + half here for video/ground-truth resolution.

    Accepts ``"<match>_1st"`` / ``"<match>_2nd"``. A bare ``"<match>"`` (no half
    suffix) is rejected with a clear error rather than guessing a half, because
    the predictions and ground truth are keyed per half.
    """
    s = str(seq)
    base, sep, suffix = s.rpartition("_")
    if not sep or suffix not in _SUFFIX_HALF:
        raise ValueError(
            f"test sequence {s!r} must end in a half suffix "
            f"({'/'.join(_HALF_SUFFIX.values())}), e.g. '117099_1st'. "
            "List per-half sequence names in cfg.data.test_matches."
        )
    return base, _SUFFIX_HALF[suffix]


# --------------------------------------------------------------------------- #
# MOTChallenge prediction writer (the only logic original to MOT)             #
# --------------------------------------------------------------------------- #

def detections_to_mot_rows(detections: Sequence[Detection]) -> list[str]:
    """Convert tracker :class:`Detection`\\ s to MOTChallenge ``data.txt`` rows.

    Pure and dependency-free (only stdlib) so it is directly unit-testable on
    synthetic detections without torch/ultralytics/cv2.

    Returns a list of comma-separated 9-column strings (no trailing newline),
    sorted by ``(frame, id)`` as MOTChallenge files conventionally are. The
    per-half tracker frame index (0-indexed) is converted to MOTChallenge's
    **1-indexed** frame by adding 1.

    Column order (docs/format-mot.md, byte-for-byte MOTChallenge)::

        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    rows: list[tuple[int, int, str]] = []
    for d in detections:
        frame = int(d.frame) + 1  # 0-indexed tracker frame -> 1-indexed MOT frame
        track_id = int(d.track_id)
        # %g keeps integer pixel coords compact (e.g. "1840") while preserving
        # sub-pixel boxes if the tracker emits them; confidence as a plain float.
        line = (
            f"{frame},{track_id},"
            f"{d.bb_left:g},{d.bb_top:g},{d.bb_width:g},{d.bb_height:g},"
            f"{float(d.conf):.6g},{_MOT_CLASS_PERSON},{_MOT_VISIBILITY}"
        )
        rows.append((frame, track_id, line))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [line for _, _, line in rows]


def write_mot_predictions(
    detections: Sequence[Detection],
    pred_root: str | Path,
    seq_name: str,
    tracker_name: str = "bytetrack",
) -> Path:
    """Write MOTChallenge predictions for one sequence.

    Layout (the native MOTChallenge ``trackers/<tracker>/data/<seq>.txt``
    convention that TrackEval's ``MotChallenge2DBox`` expects, and that
    :mod:`src.evaluation.mot_hota` is configured for)::

        <pred_root>/<tracker_name>/data/<seq_name>.txt

    The per-tracker ``data`` sub-folder and the ``<seq_name>.txt`` filename are
    not cosmetic: with ``SKIP_SPLIT_FOL=True`` (set by the evaluator) TrackEval
    reads trackers at ``<pred_root>/<tracker>/<TRACKER_SUB_FOLDER>/<seq>.txt`` with
    ``TRACKER_SUB_FOLDER='data'`` by default, and auto-discovers ``<tracker>`` by
    listing ``<pred_root>``. Verified against the installed TrackEval (see
    docs/format-mot.md "Evaluation"). Returns the written path.

    An empty ``detections`` list still writes an (empty) ``<seq>.txt`` so the
    sequence is present for the evaluator rather than silently missing.
    """
    pred_root = Path(pred_root)
    out_dir = pred_root / str(tracker_name) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_name}.txt"
    rows = detections_to_mot_rows(detections)
    # Trailing newline iff non-empty (standard for line-oriented MOT files).
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return out_path


# --------------------------------------------------------------------------- #
# Per-half driver (track stage) — needs heavy deps + data at call time        #
# --------------------------------------------------------------------------- #

def predict_sequence(
    cfg: dict,
    weights_path: str | Path,
    match_id: str,
    half: int,
) -> Path:
    """Track one test half and write its MOTChallenge ``data.txt``.

    Reuses :func:`baselines.gsr.train.run_tracker` for the actual ByteTrack pass,
    then converts to MOTChallenge rows. The heavy ultralytics import lives inside
    ``run_tracker``, so this function is import-safe; it only needs the deps when
    actually called on a video.
    """
    vpath = video_path(cfg, match_id, half)
    if not vpath.exists():
        raise FileNotFoundError(f"Test video not found: {vpath}")

    detections = run_tracker(cfg, weights_path, vpath)
    pred_root = Path(cfg["eval"].get("pred_root", "outputs/mot_baseline/preds"))
    # Tracker run dir name (the TrackEval "tracker" level). Defaults to the
    # configured tracker name so multiple runs (bytetrack/botsort/...) coexist
    # under pred_root and the evaluator can auto-discover or name them.
    tracker_name = cfg.get("tracker", {}).get("name", "bytetrack")
    return write_mot_predictions(
        detections, pred_root, sequence_name(match_id, half), tracker_name=tracker_name
    )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

_ALL_STAGES = ("dataset", "detect", "track")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="MOT baseline pipeline (SoccerTrack v2): YOLOv8-s + ByteTrack -> MOTChallenge preds."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        choices=_ALL_STAGES,
        help=(
            "Subset of stages to run, in order: "
            f"{', '.join(_ALL_STAGES)}. Default: all. "
            "'dataset' builds the YOLO dataset, 'detect' fine-tunes YOLOv8-s, "
            "'track' runs ByteTrack on the test halves and writes MOTChallenge preds."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Detector weights for the track stage (overrides cfg.detector.weights).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stages = tuple(args.stages) if args.stages else _ALL_STAGES
    print(f"[mot] config: {args.config}")
    print(f"[mot] detector: {cfg['detector']['name']}  tracker: {cfg['tracker']['name']}")
    print(f"[mot] stages: {stages}")

    data_yaml: Optional[Path] = None
    weights: Optional[Path] = args.weights or (
        Path(cfg["detector"]["weights"]) if cfg["detector"].get("weights") else None
    )

    if "dataset" in stages:
        print("[mot] stage 1/3: build YOLO dataset from MOT annotations")
        created = build_yolo_dataset(cfg)  # reused from baselines.gsr.train
        if created:
            data_yaml = created[0] / "data.yaml"
        print(f"[mot]   built {len(created)} per-half datasets")

    if "detect" in stages:
        print("[mot] stage 2/3: fine-tune YOLOv8-s detector")
        if data_yaml is None:
            ddir = Path(cfg["detector"].get("dataset_dir", "outputs/mot_baseline/yolo_dataset"))
            data_yaml = next(ddir.glob("*/data.yaml"), None)
        if data_yaml is None or not Path(data_yaml).exists():
            raise SystemExit(
                "[mot] no dataset data.yaml found; run the 'dataset' stage first "
                "or set detector.dataset_dir."
            )
        weights = train_detector(cfg, data_yaml)  # reused from baselines.gsr.train
        print(f"[mot]   weights: {weights}")

    if "track" in stages:
        if weights is None or not Path(weights).exists():
            raise SystemExit(
                "[mot] detector weights required for the 'track' stage. Pass "
                "--weights or set detector.weights / run the 'detect' stage."
            )
        print("[mot] stage 3/3: ByteTrack on each test half -> MOTChallenge preds")
        for seq in cfg["data"]["test_matches"]:
            match_id, half = parse_test_sequence(seq)
            out = predict_sequence(cfg, weights, match_id, half)
            print(f"[mot]   wrote {out}  (seq {sequence_name(match_id, half)})")

    print("[mot] done.")


if __name__ == "__main__":
    main()

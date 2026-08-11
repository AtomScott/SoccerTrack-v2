"""GS-HOTA evaluator for SoccerTrack v2 GSR predictions.

GS-HOTA is the SoccerNet Game State Reconstruction variant of HOTA: it scores
how well a method localises athletes in pitch coordinates (Euclidean distance,
5 m tolerance) *and* identifies them by role + team + jersey number (exact
match). See https://github.com/SoccerNet/sn-gamestate and the
`sn-trackeval <https://github.com/SoccerNet/sn-trackeval>`_ fork.

This module does **not** reimplement the metric. It:

  1. Converts SoccerTrack v2's on-disk GSR format (one JSON array of flat
     records per half per match — see ``docs/format-gsr.md``) into the SoccerNet
     GSR ``Labels-GameState.json`` layout the upstream scorer consumes
     (``convert_to_soccernet_gs``).
  2. Calls the upstream scorer through **one** clearly-isolated function
     (``run_upstream_gs_hota``) so the dependency surface is a single, swappable
     seam.

    python -m src.evaluation.gs_hota --pred PRED_ROOT --gt GT_ROOT --matches 128057 132831

Why the upstream call is isolated behind one function
-----------------------------------------------------
The SoccerNet GSR metric is *not* exposed as a stable
``score_match(pred_dir, gt_dir)`` Python function. It runs through the
``sn-trackeval`` fork's ``SoccerNetGS`` TrackEval dataset (driven by
``scripts/run_soccernet_gs.py``) or via the TrackLab/Hydra pipeline
(``tracklab -cn soccernet``). Both expect a *folder of per-sequence JSON files*,
not a pair of directories. Because that toolkit is an optional, heavyweight
dependency that is not installed here, ``run_upstream_gs_hota`` is the single
place that touches it: it raises an actionable ``RuntimeError`` when the toolkit
is absent, and ``convert_to_soccernet_gs`` does the real format work regardless
so a submission can run the conversion + scorer manually.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# SoccerTrack v2 -> SoccerNet GSR format conversion                            #
# --------------------------------------------------------------------------- #

# SoccerNet GSR category ids (Labels-GameState.json categories block).
# Roles map onto a single "person"-like category with a `role` attribute in
# SoccerNet's schema; we expose explicit ids so the conversion is self-describing
# and round-trippable.
# These ids mirror the categories block of the RELEASED ground truth
# (verified against $DATA/production/gsr/*/*.json): 1 player, 2 goalkeeper,
# 3 referee, 4 ball. An earlier version of this list had 4=other and 5=ball,
# which disagreed with every real file. Pitch-space GS-HOTA keys off
# attributes.role rather than category_id, so the mismatch never changed a score
# — but predictions should still describe themselves the way the GT does.
_SN_CATEGORIES = [
    {"id": 1, "name": "player", "supercategory": "object"},
    {"id": 2, "name": "goalkeeper", "supercategory": "object"},
    {"id": 3, "name": "referee", "supercategory": "object"},
    {"id": 4, "name": "ball", "supercategory": "object"},
]
_ROLE_TO_CATEGORY_ID = {c["name"]: c["id"] for c in _SN_CATEGORIES}
# Unknown/other roles fall back to "player". The released GSR ground truth only
# ever contains player and goalkeeper, so this is the safe default.
_ROLE_TO_CATEGORY_ID["other"] = 1

# SoccerNet uses centre-origin metric pitch coordinates, same frame as
# SoccerTrack v2's GSR format (docs/format-gsr.md), so x/y pass through unchanged.

# The released convention: image_id is a STRING — "3" followed by the 1-based
# frame number zero-padded to six digits — so SoccerTrack frame 0 (0-based) maps
# to "3000001". The leading digit is SoccerNet's split id (test == 3). Sequences
# staged for TrackLab instead use the full 10-character form (split id +
# 3-digit sequence id + 6-digit frame), which is why callers should pass a
# mapping derived from the ground truth whenever they have one.
_RELEASED_SPLIT_ID = "3"


def _default_image_id_for_frame(frame: int) -> str:
    """0-based SoccerTrack frame index -> released-convention string image_id."""
    return f"{_RELEASED_SPLIT_ID}{frame + 1:06d}"


def image_id_mapper_from_gt(gt_labels_path: Path) -> Callable[[int], str]:
    """Build a frame -> image_id mapping from a ground-truth GameState file.

    Prefer this over the default: it cannot drift from what the ground truth
    actually uses. The 0-based SoccerTrack frame index selects positionally from
    the frame-ordered id list, which matches the alignment verified for this
    dataset (image_id "3{N:06d}" is 1-based frame N of the panorama video).

    Only the ``images`` block is needed, but these files are ~2.7 GB so this
    still costs a full parse and roughly 15 GB of peak memory.
    """
    data = json.loads(Path(gt_labels_path).read_text())
    # Order frames exactly the way the scorer does, so positional lookup here and
    # timestep assignment there cannot disagree:
    #   soccernet_gs.py: get_frame_number = int(image['image_id'].split('_')[-1])
    ids = sorted(
        (im["image_id"] for im in data["images"]),
        key=lambda s: int(str(s).split("_")[-1]),
    )
    del data

    def _map(frame: int) -> str:
        if 0 <= frame < len(ids):
            return ids[frame]
        # Out of range: return the fallback form rather than raising, so a
        # prediction running past the annotated range is skipped by the scorer
        # instead of aborting the whole evaluation.
        return _default_image_id_for_frame(frame)

    return _map


def _half_suffix(half: int) -> str:
    if half == 1:
        return "1st"
    if half == 2:
        return "2nd"
    raise ValueError(f"half must be 1 or 2, got {half!r}")


def _sequence_name(match_id: str, half: int) -> str:
    """Per-half sequence name in the SoccerNet layout (one sequence per half)."""
    return f"{match_id}-{_half_suffix(half)}"


def soccertrack_records_to_gs(
    records: list[dict],
    seq_name: str,
    image_id_for_frame: Optional[Callable[[int], str]] = None,
) -> dict:
    """Convert one half's flat SoccerTrack GSR records to a SoccerNet GS dict.

    Record fields: ``image_id`` (0-based panoramic frame index), ``track_id``,
    ``player_id``, ``role``, ``jersey_number``, ``team_side`` ("left"/"right"),
    ``x``, ``y`` (centre-origin metres), ``bbox_image`` ``[x,y,w,h]``,
    ``bbox_pitch`` ``[x,y,w,h]``.

    This flat layout is what *predictions* look like. It is NOT what the released
    ground truth looks like — those files are already SoccerNet GameState; see
    ``_is_already_gamestate`` and docs/format-gsr.md.

    ``image_id_for_frame`` maps a 0-based frame index to the string ``image_id``
    the ground truth uses. This matters because the upstream scorer builds its
    frame index ONLY from the ground truth's ``images`` list, so a prediction
    whose ``image_id`` is absent from it matches nothing and is silently dropped.
    Two further traps in the same code path: given an integer ``image_id`` the
    scorer's unmatched-id branch calls ``len()`` on an int and raises TypeError,
    and only 10-character ids take its clean skip path.

    The default reproduces the released convention (see
    ``_default_image_id_for_frame``), measured to cover 67,625 of 67,625
    prediction frames for 128057's first half. Prefer
    ``image_id_mapper_from_gt`` when a ground-truth file is at hand; the default
    is a documented fallback, not an authority.

    This is a pure function (no I/O) and is the unit-tested core of the adapter.
    """
    if image_id_for_frame is None:
        image_id_for_frame = _default_image_id_for_frame

    images: dict[str, dict] = {}
    annotations: list[dict] = []

    for i, r in enumerate(records):
        frame = int(r["image_id"])            # 0-based SoccerTrack frame index
        image_id = image_id_for_frame(frame)
        if image_id not in images:
            images[image_id] = {
                "image_id": image_id,
                # SoccerNet file names are 1-based: frame 0 -> 000001.jpg.
                "file_name": f"{frame + 1:06d}.jpg",
                "frame_idx": frame,
            }

        role = r.get("role", "player")
        category_id = _ROLE_TO_CATEGORY_ID.get(role, _ROLE_TO_CATEGORY_ID["other"])

        ann: dict = {
            "id": i + 1,
            "image_id": image_id,
            "track_id": int(r["track_id"]),
            "category_id": category_id,
            "supercategory": "object",
            "attributes": {
                "role": role,
                "team": r.get("team_side"),          # "left"/"right"/None
                "jersey": _as_jersey(r.get("jersey_number")),
            },
        }

        # Image-plane bbox: [x, y, w, h] (ints) -> dict form expected by SoccerNet.
        bi = r.get("bbox_image")
        if bi is not None:
            ann["bbox_image"] = {
                "x": float(bi[0]),
                "y": float(bi[1]),
                "w": float(bi[2]),
                "h": float(bi[3]),
                "x_center": float(bi[0]) + float(bi[2]) / 2.0,
                "y_center": float(bi[1]) + float(bi[3]) / 2.0,
            }

        # Pitch position: SoccerNet keys the localisation similarity off the
        # bottom-middle pitch point. SoccerTrack `x`/`y` are exactly that
        # (centre-origin metres, the player's feet — see docs/format-gsr.md).
        #
        # All SIX keys are required. With EVAL_SPACE='pitch' the scorer reads
        # x/y_bottom_left, x/y_bottom_middle and x/y_bottom_right unconditionally
        # (trackeval/datasets/soccernet_gs.py) and raises KeyError on a subset.
        # In the released ground truth these three points are DEGENERATE — all
        # three hold identical values and only the middle is informative — so
        # replicating the middle point across all three is faithful to how the GT
        # is built, not an invention on our part.
        _px, _py = float(r["x"]), float(r["y"])
        ann["bbox_pitch"] = {
            "x_bottom_left": _px, "y_bottom_left": _py,
            "x_bottom_middle": _px, "y_bottom_middle": _py,
            "x_bottom_right": _px, "y_bottom_right": _py,
        }
        bp = r.get("bbox_pitch")
        if bp is not None:
            ann["bbox_pitch"].update(
                {
                    "x": float(bp[0]),
                    "y": float(bp[1]),
                    "w": float(bp[2]),
                    "h": float(bp[3]),
                }
            )
        annotations.append(ann)

    return {
        "info": {
            "version": "1.3",
            "name": seq_name,
            "source": "soccertrack-v2",
        },
        "images": [images[k] for k in sorted(images)],
        "annotations": annotations,
        "categories": _SN_CATEGORIES,
    }


def _as_jersey(v) -> Optional[str]:
    """SoccerNet stores jersey as a string ("9") or None when unobserved."""
    if v is None or v == "":
        return None
    return str(int(v))


def convert_to_soccernet_gs(
    src_root: Path,
    dst_root: Path,
    match_ids: list[str],
    tracker_name: str = "soccertrack",
    is_gt: bool = True,
) -> Path:
    """Convert a SoccerTrack GSR tree into a SoccerNet GSR folder, on disk.

    Reads ``src_root/<match>/<match>_{1st,2nd}.json`` (the SoccerTrack layout used
    by predictions *and* ground truth) and writes, per half, a SoccerNet
    ``Labels-GameState.json`` under::

        dst_root/<seq_name>/Labels-GameState.json        # for ground truth
        dst_root/<tracker_name>/<seq_name>/Labels-GameState.json  # for predictions

    The exact subfolder convention of the upstream scorer differs between GT and
    tracker outputs; ``run_upstream_gs_hota`` wires the two together. Returns
    ``dst_root``.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    for match_id in match_ids:
        for half in (1, 2):
            src = src_root / str(match_id) / f"{match_id}_{_half_suffix(half)}.json"
            if not src.exists():
                # Halves may be missing in partial snapshots; skip explicitly.
                continue
            seq = _sequence_name(str(match_id), half)
            # The scorer's two sides use DIFFERENT layouts (verified against
            # sn-trackeval 0.4.0's SoccerNetGS.__init__):
            #   ground truth : <root>/<seq>/Labels-GameState.json   (a dir per sequence)
            #   predictions  : <root>/<tracker>/data/<seq>.json      (a flat file)
            # and it reads GT from the "annotations" key but predictions from
            # "predictions" (soccernet_gs.py: key = "annotations" if is_gt else
            # "predictions"). Getting either wrong yields "file not found" or a silent
            # zero rather than a useful error.
            if is_gt:
                dst = dst_root / seq / "Labels-GameState.json"
            else:
                dst = dst_root / tracker_name / "data" / f"{seq}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)

            if _is_already_gamestate(src):
                # The released production tree is ALREADY in SoccerNet
                # Labels-GameState.json form, so there is nothing to convert. Parsing it
                # would be both wrong and ruinous: these files are ~2.7 GB per half, so
                # json.loads needs roughly 20 GB of RAM, and soccertrack_records_to_gs
                # expects a flat list and raises TypeError on a dict.
                #
                # It is also the only form that satisfies the pitch-space scorer, which
                # requires six bbox_pitch keys (x/y_bottom_left, _middle, _right) whereas
                # soccertrack_records_to_gs emits only _middle. So linking is not merely
                # an optimisation -- re-converting would trip the scorer's assert.
                if is_gt or _has_predictions_key(src):
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    try:
                        dst.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, dst)  # filesystems without symlink support
                    continue
                raise ValueError(
                    f"{src} is in Labels-GameState form but stores its detections under "
                    "'annotations'. The scorer reads predictions from a 'predictions' key, "
                    "so this file cannot be used as a tracker output as-is. Emit "
                    "predictions either as the flat record list documented in "
                    "docs/format-gsr.md, or as Labels-GameState with a 'predictions' key."
                )

            records = json.loads(src.read_text())
            gs = soccertrack_records_to_gs(records, seq)
            if not is_gt:
                gs["predictions"] = gs.pop("annotations")
            dst.write_text(json.dumps(gs, indent=2))
    return dst_root


def _has_predictions_key(path: Path, probe_bytes: int = 4_000_000) -> bool:
    """Whether a Labels-GameState file stores detections under "predictions".

    Only the head is probed so gigabyte files are not read. "images" precedes the
    detections block in every file produced here, so a few MB is ample.
    """
    with open(path, "rb") as fh:
        return b'"predictions"' in fh.read(probe_bytes)


def _is_already_gamestate(path: Path) -> bool:
    """True when *path* is a SoccerNet Labels-GameState dict rather than a flat record list.

    Decided from the first non-whitespace byte so that a multi-gigabyte file is never read
    into memory: ``{`` is the COCO-style dict SoccerNet expects, ``[`` is the flat
    SoccerTrack record list documented in docs/format-gsr.md.
    """
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64)
            if not chunk:
                return False
            stripped = chunk.lstrip()
            if stripped:
                return stripped[:1] == b"{"


# --------------------------------------------------------------------------- #
# Public scoring entry points                                                  #
# --------------------------------------------------------------------------- #

def score_many(
    pred_root: Path,
    gt_root: Path,
    match_ids: list[str],
    workdir: Optional[Path] = None,
) -> dict:
    """Score ``match_ids`` with GS-HOTA. Returns ``{seq: metrics, "overall": ...}``.

    Steps:
      1. Convert both ``pred_root`` and ``gt_root`` (SoccerTrack GSR layout) into
         the SoccerNet GSR layout under ``workdir`` (a temp dir if unset).
      2. Hand the converted folders to :func:`run_upstream_gs_hota`.

    Requires the SoccerNet GSR toolkit (``sn-trackeval``) for step 2; raises an
    actionable ``RuntimeError`` otherwise. Step 1 always runs so a submitter can
    inspect / score the converted folders manually.
    """
    import tempfile

    pred_root = Path(pred_root)
    gt_root = Path(gt_root)
    mids = [str(m) for m in match_ids]

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="gs_hota_"))
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    gt_dst = convert_to_soccernet_gs(gt_root, workdir / "gt", mids, is_gt=True)
    pred_dst = convert_to_soccernet_gs(
        pred_root, workdir / "preds", mids, tracker_name="soccertrack", is_gt=False
    )
    return run_upstream_gs_hota(pred_dst, gt_dst, mids, tracker_name="soccertrack")


def run_upstream_gs_hota(
    pred_gs_root: Path,
    gt_gs_root: Path,
    match_ids: list[str],
    tracker_name: str = "soccertrack",
) -> dict:
    """THE single seam onto the upstream SoccerNet GS-HOTA scorer.

    ``pred_gs_root`` / ``gt_gs_root`` are folders already in SoccerNet GSR layout
    (as produced by :func:`convert_to_soccernet_gs`). This function is the only
    place in the codebase that imports / invokes the upstream metric.

    The upstream metric lives in the ``sn-trackeval`` fork
    (https://github.com/SoccerNet/sn-trackeval) as a ``SoccerNetGS`` TrackEval
    dataset, normally run via its ``scripts/run_soccernet_gs.py`` or the TrackLab
    pipeline (``tracklab -cn soccernet``). There is **no** stable importable
    ``score_match`` Python API, so callers running a real evaluation should point
    the fork's runner at these converted folders, e.g.::

        python sn-trackeval/scripts/run_soccernet_gs.py \\
            --GT_FOLDER  <gt_gs_root> \\
            --TRACKERS_FOLDER <pred_gs_root> \\
            --TRACKER_SUB_FOLDER soccertrack

    When the toolkit is importable we drive it programmatically; otherwise we
    raise with install guidance rather than silently returning fake numbers.
    """
    try:
        import trackeval  # type: ignore  # noqa: F401
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "The SoccerNet GS-HOTA scorer (sn-trackeval) is not installed, so the "
            "metric cannot be computed here. The SoccerTrack->SoccerNet conversion "
            f"succeeded and converted folders are at:\n  GT:    {gt_gs_root}\n  PRED:  "
            f"{pred_gs_root}\nInstall the SoccerNet TrackEval fork "
            "(https://github.com/SoccerNet/sn-trackeval) and run its "
            "scripts/run_soccernet_gs.py against those folders, or run the full "
            "TrackLab pipeline (`tracklab -cn soccernet`)."
        ) from e

    # Drive the SoccerNetGS dataset directly. Kept behind this single seam so it is the
    # only thing to update if the fork's API shifts.
    #
    # VERIFIED against sn-trackeval 0.4.0 (see tests/test_gs_hota_identity.py, which
    # scores a real production half against itself and asserts GS-HOTA == 1.0). The four
    # non-default keys below are each load-bearing:
    #
    #   SKIP_SPLIT_FOL=True   Default False makes the dataset look under
    #                         <GT_FOLDER>/valid/<seq>/ and
    #                         <TRACKERS_FOLDER>/SoccerNetGS-valid/<tracker>/. Our layout
    #                         has no split folder, so GT would not be found.
    #   SEQ_INFO              Without it, sequence discovery lists
    #                         <GT_FOLDER>/<SPLIT_TO_EVAL> -- note it uses GT_FOLDER and
    #                         not the split-stripped gt_fol, so it looks for a 'valid'
    #                         directory even when SKIP_SPLIT_FOL is True and raises
    #                         "No sequences are selected to be evaluated."
    #   TRACKER_SUB_FOLDER    Predictions resolve to
    #                         <TRACKERS_FOLDER>/<tracker>/<sub>/<seq>.json.
    #   EVAL_SPACE='pitch'    SoccerTrack v2 has NO ground-truth detections, so image
    #                         space is not scoreable. Pitch space is also the scorer's
    #                         default; it is pinned explicitly so a future default change
    #                         cannot silently start reading bounding boxes.
    import trackeval as te  # type: ignore

    datasets_mod = getattr(te, "datasets", None)
    SoccerNetGS = getattr(datasets_mod, "SoccerNetGS", None) if datasets_mod else None
    if SoccerNetGS is None:
        raise RuntimeError(
            "Installed TrackEval lacks the SoccerNetGS dataset (need the SoccerNet "
            "fork, https://github.com/SoccerNet/sn-trackeval). Converted folders "
            f"are at GT={gt_gs_root}, PRED={pred_gs_root}; run the fork's "
            "scripts/run_soccernet_gs.py against them."
        )

    eval_cfg = te.Evaluator.get_default_eval_config()
    eval_cfg.update(
        {
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "PRINT_RESULTS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "USE_PARALLEL": False,
        }
    )
    seqs = [_sequence_name(str(m), h) for m in match_ids for h in (1, 2)
            if (gt_gs_root / _sequence_name(str(m), h) / "Labels-GameState.json").exists()]
    if not seqs:
        raise RuntimeError(
            f"No converted sequences found under {gt_gs_root}. Expected "
            f"<seq>/Labels-GameState.json for match ids {match_ids}."
        )

    dataset_cfg = SoccerNetGS.get_default_dataset_config()
    dataset_cfg.update(
        {
            "GT_FOLDER": str(gt_gs_root),
            "TRACKERS_FOLDER": str(pred_gs_root),
            "TRACKERS_TO_EVAL": [tracker_name],
            "TRACKER_SUB_FOLDER": "data",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {s: None for s in seqs},
            "EVAL_SPACE": "pitch",
            "OUTPUT_FOLDER": None,
            "PRINT_CONFIG": False,
        }
    )
    evaluator = te.Evaluator(eval_cfg)
    results, _ = evaluator.evaluate([SoccerNetGS(dataset_cfg)], [te.metrics.HOTA()])
    return _flatten_results(results)


def _scalar(value) -> float:
    """Reduce a TrackEval metric value to a single float.

    TrackEval reports HOTA-family metrics (``HOTA``, ``DetA``, ``AssA``, ``LocA``,
    ...) as numpy arrays over the alpha (localisation-threshold) axis; the
    conventionally-reported scalar is the **mean** over that axis. Scalar fields
    (e.g. ``HOTA(0)``) and python numbers pass straight through.
    """
    import numpy as np  # local import: keep module import cheap / numpy-optional

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _hota_field_dict(cls_metrics: dict) -> Optional[dict]:
    """Locate the HOTA *field* dict inside one class's TrackEval result.

    TrackEval nests one extra level per the upstream contract
    (``res[seq][class][metric_name][field]``, e.g.
    ``res[seq][pedestrian][HOTA][DetA]``): a class's dict is keyed by metric-group
    name (``HOTA``/``Identity``/``CLEAR``/``Count``), and the HOTA *fields*
    (the ``HOTA`` array, ``DetA`` ...) live under the ``HOTA`` group.

    Returns that field dict, tolerating two shapes:
      * grouped (real TrackEval): ``{"HOTA": {"HOTA": array, "DetA": ...}, ...}``
        -> returns ``cls_metrics["HOTA"]``;
      * already-flat (defensive): ``{"HOTA": array, "DetA": ...}`` -> returns
        ``cls_metrics`` itself.
    Returns ``None`` if no HOTA fields are found.
    """
    if not isinstance(cls_metrics, dict):
        return None
    group = cls_metrics.get("HOTA")
    # Grouped form: the "HOTA" value is itself a dict of fields containing "HOTA".
    if isinstance(group, dict) and "HOTA" in group:
        return group
    # Flat form: the "HOTA" value is the array/number directly.
    if group is not None and not isinstance(group, dict):
        return cls_metrics
    # Last resort: scan any sub-dict that exposes a "HOTA" field.
    for v in cls_metrics.values():
        if isinstance(v, dict) and "HOTA" in v and not isinstance(v["HOTA"], dict):
            return v
    return None


def _normalize_metrics(cls_metrics: dict) -> dict:
    """Normalize one class's TrackEval metric dict so callers get scalars.

    Descends to the HOTA field dict (see :func:`_hota_field_dict`) whose keys are
    the HOTA-family metrics (``HOTA``, ``DetA``, ``AssA``, ``LocA`` — each a numpy
    array over the alpha axis). This:

      * adds a scalar ``GS-HOTA`` (mean of the ``HOTA`` array) — the key
        ``baselines/gsr/eval.py`` reads as ``scores['overall']['GS-HOTA']``;
      * adds scalar ``HOTA`` / ``DetA`` / ``AssA`` / ``LocA`` (the standard paper
        breakdown) as JSON-serialisable floats;
      * preserves the original (unflattened) class dict under ``_raw`` so the full
        per-alpha arrays and other metric groups (Identity/CLEAR/Count) are not
        lost.

    GS-HOTA *is* HOTA computed under the SoccerNetGS dataset's pitch-distance +
    identity similarity, so the reported scalar is the mean of the HOTA-over-alpha
    array — identical reduction to standard HOTA, different similarity upstream.
    """
    if not isinstance(cls_metrics, dict):
        return cls_metrics

    fields = _hota_field_dict(cls_metrics)
    out: dict = {"_raw": cls_metrics}
    if fields is not None:
        for name in ("HOTA", "DetA", "AssA", "LocA"):
            if name in fields:
                out[name] = _scalar(fields[name])
        if "HOTA" in out:
            out["GS-HOTA"] = out["HOTA"]
    return out


def _flatten_results(results: dict) -> dict:
    """Best-effort flatten of TrackEval's nested results into {seq: metrics}.

    Returns ``{seq: normalized_metrics, "overall": normalized_metrics}`` where each
    ``normalized_metrics`` exposes a scalar ``GS-HOTA`` (plus HOTA/DetA/AssA/LocA)
    via :func:`_normalize_metrics`. ``overall`` is TrackEval's ``COMBINED_SEQ``.

    NOTE (unverified): the SoccerNetGS class name under which HOTA is reported is
    not necessarily ``"pedestrian"``; we take that class if present and otherwise
    fall back to the first class. This and the upstream config wiring in
    :func:`run_upstream_gs_hota` are exercised only against the real
    ``sn-trackeval`` fork (not installed here); the *normalization shape* is
    verified against stock TrackEval's HOTA output.
    """
    flat: dict[str, dict] = {}
    # TrackEval nests: results[dataset][tracker]['COMBINED_SEQ'/seq][class][metric].
    try:
        for _ds, trackers in results.items():
            for _tracker, seqs in trackers.items():
                for seq, classes in seqs.items():
                    # Take the pedestrian/person class if present, else first.
                    cls = classes.get("pedestrian") or next(iter(classes.values()))
                    flat[seq] = _normalize_metrics(cls)
    except (AttributeError, StopIteration):
        return results
    overall = flat.pop("COMBINED_SEQ", None)
    if overall is not None:
        flat["overall"] = overall
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Score GSR predictions with GS-HOTA.")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction root (SoccerTrack GSR layout).")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth gsr/ root (SoccerTrack GSR layout).")
    parser.add_argument("--matches", nargs="*", required=True, help="Match IDs to score.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Where to write the converted SoccerNet-format folders (default: temp dir).",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only run the SoccerTrack->SoccerNet conversion; skip the upstream scorer.",
    )
    args = parser.parse_args()

    mids = [str(m) for m in args.matches]
    if args.convert_only:
        workdir = args.workdir or Path("outputs/gsr_baseline/soccernet_fmt")
        gt_dst = convert_to_soccernet_gs(args.gt, workdir / "gt", mids)
        pred_dst = convert_to_soccernet_gs(args.pred, workdir / "preds", mids, tracker_name="soccertrack")
        print(f"converted GT  -> {gt_dst}")
        print(f"converted PRED -> {pred_dst}")
        return

    scores = score_many(args.pred, args.gt, mids, workdir=args.workdir)
    text = json.dumps(scores, indent=2, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

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
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# SoccerTrack v2 -> SoccerNet GSR format conversion                            #
# --------------------------------------------------------------------------- #

# SoccerNet GSR category ids (Labels-GameState.json categories block).
# Roles map onto a single "person"-like category with a `role` attribute in
# SoccerNet's schema; we expose explicit ids so the conversion is self-describing
# and round-trippable.
_SN_CATEGORIES = [
    {"id": 1, "name": "player", "supercategory": "person"},
    {"id": 2, "name": "goalkeeper", "supercategory": "person"},
    {"id": 3, "name": "referee", "supercategory": "person"},
    {"id": 4, "name": "other", "supercategory": "person"},
    {"id": 5, "name": "ball", "supercategory": "ball"},
]
_ROLE_TO_CATEGORY_ID = {c["name"]: c["id"] for c in _SN_CATEGORIES}

# SoccerNet uses centre-origin metric pitch coordinates, same frame as
# SoccerTrack v2's GSR format (docs/format-gsr.md), so x/y pass through unchanged.


def _half_suffix(half: int) -> str:
    if half == 1:
        return "1st"
    if half == 2:
        return "2nd"
    raise ValueError(f"half must be 1 or 2, got {half!r}")


def _sequence_name(match_id: str, half: int) -> str:
    """Per-half sequence name in the SoccerNet layout (one sequence per half)."""
    return f"{match_id}-{_half_suffix(half)}"


def soccertrack_records_to_gs(records: list[dict], seq_name: str) -> dict:
    """Convert one half's flat SoccerTrack GSR records to a SoccerNet GS dict.

    SoccerTrack record fields (docs/format-gsr.md): ``image_id``, ``track_id``,
    ``player_id``, ``role``, ``jersey_number``, ``team_side`` ("left"/"right"),
    ``x``, ``y`` (centre-origin metres), ``bbox_image`` ``[x,y,w,h]``,
    ``bbox_pitch`` ``[x,y,w,h]``.

    SoccerNet ``Labels-GameState.json`` is a COCO-style dict with ``info``,
    ``images``, ``annotations``, ``categories``. Each annotation carries
    ``image_id``, ``track_id``, ``category_id``, ``bbox_image``
    (``{x,y,w,h}`` dict), ``bbox_pitch`` (``{x_bottom_middle, y_bottom_middle,
    ...}``), and an ``attributes`` block with ``role``, ``team`` ("left"/"right"),
    ``jersey``.

    This is a pure function (no I/O) and is the unit-tested core of the adapter.
    """
    images: dict[int, dict] = {}
    annotations: list[dict] = []

    for i, r in enumerate(records):
        image_id = int(r["image_id"])
        if image_id not in images:
            images[image_id] = {
                "image_id": image_id,
                "file_name": f"{image_id:06d}.jpg",
                # SoccerTrack panoramic frame index doubles as the frame number.
                "frame_idx": image_id,
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
        ann["bbox_pitch"] = {
            "x_bottom_middle": float(r["x"]),
            "y_bottom_middle": float(r["y"]),
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
            records = json.loads(src.read_text())
            seq = _sequence_name(str(match_id), half)
            gs = soccertrack_records_to_gs(records, seq)
            out_dir = dst_root / seq
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "Labels-GameState.json").write_text(json.dumps(gs, indent=2))
    return dst_root


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

    gt_dst = convert_to_soccernet_gs(gt_root, workdir / "gt", mids)
    pred_dst = convert_to_soccernet_gs(
        pred_root, workdir / "preds", mids, tracker_name="soccertrack"
    )
    return run_upstream_gs_hota(pred_dst, gt_dst, mids)


def run_upstream_gs_hota(
    pred_gs_root: Path,
    gt_gs_root: Path,
    match_ids: list[str],
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

    # If a SoccerNetGS TrackEval dataset is available, drive it directly. The
    # exact config keys track the sn-trackeval fork; kept behind this single seam
    # so it is the only thing to update if the fork's API shifts.
    #
    # REGRESSION NOTE (unverified): the dataset_cfg keys below (GT_FOLDER,
    # TRACKERS_FOLDER, TRACKERS_TO_EVAL, ...) and SoccerNetGS.get_default_dataset_config()
    # are NOT exercised here — only stock TrackEval (no SoccerNetGS dataset) is
    # installed in this environment. Stock TrackEval's HOTA *output shape* is
    # verified (drives _flatten_results / _normalize_metrics); the SoccerNetGS
    # config wiring must be re-checked against a real sn-trackeval install. The
    # canonical command (docstring above) is the supported escape hatch meanwhile.
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
    eval_cfg["PRINT_CONFIG"] = False
    eval_cfg["TIME_PROGRESS"] = False
    dataset_cfg = SoccerNetGS.get_default_dataset_config()
    dataset_cfg.update(
        {
            "GT_FOLDER": str(gt_gs_root),
            "TRACKERS_FOLDER": str(pred_gs_root),
            "TRACKERS_TO_EVAL": ["soccertrack"],
            "OUTPUT_FOLDER": None,
            "PRINT_CONFIG": False,
        }
    )
    metric = te.metrics.HOTA() if hasattr(te.metrics, "HOTA") else None
    evaluator = te.Evaluator(eval_cfg)
    results, _ = evaluator.evaluate([SoccerNetGS(dataset_cfg)], [metric] if metric else [])
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

"""GS-HOTA prediction-path test: a perfect prediction must score 1.0.

WHY THIS EXISTS SEPARATELY FROM tests/test_gs_hota_identity.py
    That test scores the ground truth against itself. But the released GSR ground truth is
    ALREADY SoccerNet GameState, so `convert_to_soccernet_gs` recognises it and symlinks the
    same file in as both ground truth and tracker output. The flat-records converter is never
    called. It therefore proved the scorer's configuration and nothing about the code path a
    real prediction file takes.

    Three separate defects lived in that unexercised path, and scoring a real prediction hit
    all three in sequence:

      1. bbox_pitch carried only x/y_bottom_middle. With EVAL_SPACE='pitch' the scorer reads
         six keys unconditionally -> KeyError.
      2. image_id was an int. The scorer's unmatched-id branch does `len(image_id) == 10`
         -> TypeError: object of type 'int' has no len().
      3. predictions used their own 0-based integer frame numbering, which shares no values
         with the ground truth's string image_ids, so nothing matched.

    This test takes a slice of real ground truth, degrades it into the flat prediction layout,
    pushes it back through the converter as a PREDICTION, and scores it. A perfect prediction
    can only score 1.0 if all six pitch keys, the category_id and the image_id mapping are
    right, so any regression in the prediction path fails here.

Run:
    .venv/bin/python -m pytest tests/test_gs_hota_prediction_path.py -q -s
    .venv/bin/python tests/test_gs_hota_prediction_path.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Resolve THIS checkout's src/, not whatever the venv's editable install points at.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("DATA", "/data/share/SoccerTrack-v2/data"))
GT_ROOT = DATA / "production" / "gsr"
MATCH = os.environ.get("GS_HOTA_TEST_MATCH", "128057")
HALF = "1st"
# A slice keeps the test fast. The released halves are ~2.7 GB and parsing one costs
# minutes and ~15 GB of RAM, so we read it once and cut a window out of it.
N_FRAMES = int(os.environ.get("GS_HOTA_TEST_FRAMES", "300"))


def _gt_path() -> Path:
    return GT_ROOT / MATCH / f"{MATCH}_{HALF}.json"


def _load_slice():
    """Return (gt_gamestate_slice, flat_prediction_records) for the first N_FRAMES frames.

    The predictions are a lossless re-encoding of the ground truth into the flat layout, so a
    correct converter must reproduce a perfect match.
    """
    d = json.loads(_gt_path().read_text())
    images = sorted(d["images"], key=lambda im: int(str(im["image_id"]).split("_")[-1]))[:N_FRAMES]
    keep = {im["image_id"] for im in images}
    anns = [
        a for a in d["annotations"]
        if a.get("supercategory") == "object" and a["image_id"] in keep
    ]
    gt = {
        "info": dict(d["info"]),
        "images": images,
        "annotations": anns,
        "categories": d["categories"],
    }
    gt["info"]["seq_length"] = len(images)

    # image_id "3{N:06d}" is 1-based frame N; our flat layout is 0-based, hence the -1.
    first = int(str(images[0]["image_id"]).split("_")[-1])
    records = []
    for a in anns:
        n = int(str(a["image_id"]).split("_")[-1])
        bp = a["bbox_pitch"]
        attrs = a.get("attributes", {})
        records.append({
            "image_id": n - first,                  # 0-based within the slice
            "track_id": int(a["track_id"]),
            "role": attrs.get("role", "player"),
            "jersey_number": attrs.get("jersey"),
            "team_side": attrs.get("team"),
            "x": float(bp["x_bottom_middle"]),
            "y": float(bp["y_bottom_middle"]),
        })
    return gt, records, images


def _available() -> bool:
    return _gt_path().exists()


def test_converter_emits_what_the_scorer_reads():
    """Structural guard: the six pitch keys, a category_id, and a string image_id."""
    if not _available():
        print(f"SKIP: {_gt_path()} not present")
        return
    from src.evaluation.gs_hota import soccertrack_records_to_gs

    _gt, records, _images = _load_slice()
    gs = soccertrack_records_to_gs(records[:50], f"{MATCH}-{HALF}")
    a = gs["annotations"][0]

    required = {
        "x_bottom_left", "y_bottom_left",
        "x_bottom_middle", "y_bottom_middle",
        "x_bottom_right", "y_bottom_right",
    }
    missing = required - set(a["bbox_pitch"])
    assert not missing, f"bbox_pitch missing {sorted(missing)}; the scorer reads all six"
    assert "category_id" in a, "annotations need category_id or the scorer raises KeyError"
    assert isinstance(a["image_id"], str), (
        f"image_id must be a str, got {type(a['image_id']).__name__}; the scorer calls len() on it"
    )
    assert len(a["image_id"]) >= 7, f"unexpected image_id form: {a['image_id']!r}"
    # The three pitch points are degenerate in the ground truth; we must match that.
    bp = a["bbox_pitch"]
    assert bp["x_bottom_left"] == bp["x_bottom_middle"] == bp["x_bottom_right"]
    print(f"structure OK: image_id={a['image_id']!r} category_id={a['category_id']} "
          f"bbox_pitch keys={len(bp)}")


def test_perfect_prediction_scores_one():
    """End-to-end: ground truth re-encoded as a prediction must score HOTA ~1.0."""
    if not _available():
        print(f"SKIP: {_gt_path()} not present")
        return
    import numpy as np
    import trackeval as te
    from trackeval.datasets import SoccerNetGS

    from src.evaluation.gs_hota import soccertrack_records_to_gs

    gt, records, images = _load_slice()
    seq = f"{MATCH}-{HALF}"

    # Map our 0-based slice frames onto the ground truth's own image_ids. This is the
    # mapping a real run must get right; hardcoding it here would defeat the test, so we
    # derive it from the slice we are scoring against.
    ids = [im["image_id"] for im in images]
    gs_pred = soccertrack_records_to_gs(
        records, seq, image_id_for_frame=lambda n: ids[n] if 0 <= n < len(ids) else "0"
    )

    tmp = Path(tempfile.mkdtemp(prefix="gs-hota-pred-"))
    gt_dir = tmp / "gt" / seq
    gt_dir.mkdir(parents=True)
    (gt_dir / "Labels-GameState.json").write_text(json.dumps(gt))
    pred_dir = tmp / "pred" / "soccertrack" / "data"
    pred_dir.mkdir(parents=True)
    # The scorer reads data["predictions"] for tracker files, not data["annotations"].
    (pred_dir / f"{seq}.json").write_text(json.dumps({"predictions": gs_pred["annotations"]}))

    ec = te.Evaluator.get_default_eval_config()
    ec.update({"PRINT_CONFIG": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
               "PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
               "PLOT_CURVES": False, "USE_PARALLEL": False})
    dc = SoccerNetGS.get_default_dataset_config()
    dc.update({"GT_FOLDER": str(tmp / "gt"), "TRACKERS_FOLDER": str(tmp / "pred"),
               "TRACKERS_TO_EVAL": ["soccertrack"], "TRACKER_SUB_FOLDER": "data",
               "SKIP_SPLIT_FOL": True, "SEQ_INFO": {seq: None}, "EVAL_SPACE": "pitch",
               "OUTPUT_FOLDER": None, "PRINT_CONFIG": False,
               # Attributes off: this test is about the geometry and id plumbing, not ReID.
               "USE_ROLES": False, "USE_TEAMS": False, "USE_JERSEY_NUMBERS": False})

    res, _ = te.Evaluator(ec).evaluate([SoccerNetGS(dc)], [te.metrics.HOTA()])
    r = res["SoccerNetGS"]["soccertrack"][seq]["person"]["HOTA"]
    c = res["SoccerNetGS"]["soccertrack"][seq]["person"]["Count"]
    hota = float(np.mean(np.atleast_1d(r["HOTA"])))
    loca = float(np.mean(np.atleast_1d(r["LocA"])))
    print(f"HOTA={hota:.6f} LocA={loca:.6f}  "
          f"pred dets={c['Dets']} GT dets={c['GT_Dets']} frames={c['Frames']}")
    assert c["Dets"] == c["GT_Dets"], (
        f"prediction lost detections: {c['Dets']} vs {c['GT_Dets']} — the image_id mapping "
        "is dropping frames"
    )
    assert hota > 0.999, (
        f"a perfect prediction scored HOTA {hota}, not ~1.0. Do not relax this threshold; "
        "it means the prediction path is misencoding geometry or frame ids."
    )


if __name__ == "__main__":
    test_converter_emits_what_the_scorer_reads()
    test_perfect_prediction_scores_one()
    print("PASS")

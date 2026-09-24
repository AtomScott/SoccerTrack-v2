"""Rescore one whole-half prediction file on nested prefixes of the half.

For a half that was run end to end, score the SAME predictions against the
ground truth over the first N frames only, for several N. This measures how
the score of one run changes with the evaluation horizon, holding the
predictions fixed; it is NOT a rerun of the pipeline on shorter footage
(that is stage_prefix_sequences.py plus a pipeline run, done for 128057
first half only).

Prediction and GT entries are kept when their image_id belongs to the first
N images of the GT (images sorted by file_name), so a prefix is exactly the
opening N frames of the annotated half. Non-finite pitch coordinates are
dropped as in score_one.py, and both scorings (attributes on and off) are
computed with the same trackeval configuration.

    python scripts/gsr/score_prefixes.py --pred <pred.json> --gt <Labels-GameState.json> \
        --out results/gsr/prefix_rescoring/<seq>.json [--frames 750 1500 3000 7500 15000 30000]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import trackeval as te
from trackeval.datasets import SoccerNetGS


def finite_pitch(entry: dict) -> bool:
    bp = entry.get("bbox_pitch")
    if not bp:
        return False
    for k in ("x_bottom_middle", "y_bottom_middle"):
        v = bp.get(k)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return False
    return True


def clean(entries):
    kept, dropped = [], 0
    for e in entries:
        if e.get("supercategory") == "object" and not finite_pitch(e):
            dropped += 1
            continue
        kept.append(e)
    return kept, dropped


def score(seq, gt, pred):
    tmp = Path(tempfile.mkdtemp(prefix="gshota_pref_"))
    try:
        gt_root = tmp / "gt" / seq
        gt_root.mkdir(parents=True)
        (gt_root / "Labels-GameState.json").write_text(json.dumps(gt))
        pr_root = tmp / "pred" / "t" / "data"
        pr_root.mkdir(parents=True)
        (pr_root / f"{seq}.json").write_text(json.dumps(pred))
        ec = te.Evaluator.get_default_eval_config()
        ec.update({"PRINT_CONFIG": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
                   "PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                   "PLOT_CURVES": False, "USE_PARALLEL": False})
        rows = {}
        for label, attrs in (("attributes ON (official GS-HOTA)", True),
                             ("attributes OFF (geometry + association)", False)):
            dc = SoccerNetGS.get_default_dataset_config()
            dc.update({"GT_FOLDER": str(tmp / "gt"), "TRACKERS_FOLDER": str(tmp / "pred"),
                       "TRACKERS_TO_EVAL": ["t"], "TRACKER_SUB_FOLDER": "data",
                       "SKIP_SPLIT_FOL": True, "SEQ_INFO": {seq: None}, "EVAL_SPACE": "pitch",
                       "OUTPUT_FOLDER": None, "PRINT_CONFIG": False,
                       "USE_ROLES": attrs, "USE_TEAMS": attrs, "USE_JERSEY_NUMBERS": attrs})
            res, _ = te.Evaluator(ec).evaluate([SoccerNetGS(dc)], [te.metrics.HOTA()])
            h = res["SoccerNetGS"]["t"][seq]["person"]["HOTA"]
            c = res["SoccerNetGS"]["t"][seq]["person"]["Count"]
            g = lambda k: float(sum(list(h[k]) if hasattr(h[k], "__len__") else [h[k]]) /
                                max(1, len(h[k]) if hasattr(h[k], "__len__") else 1)) * 100
            rows[label] = {k: g(k) for k in ("HOTA", "DetA", "AssA", "LocA")}
            rows[label].update({"ids": int(c["IDs"]), "gt_ids": int(c["GT_IDs"]),
                                "dets": int(c["Dets"]), "gt_dets": int(c["GT_Dets"])})
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[750, 1500, 3000, 7500, 15000, 30000])
    a = ap.parse_args()
    seq = a.seq or a.gt.parent.name

    t0 = time.time()
    gt = json.loads(a.gt.read_text())
    pred = json.loads(a.pred.read_text())
    print(f"{seq}: loaded GT ({len(gt['images']):,} images) and predictions "
          f"({len(pred['predictions']):,} entries) in {time.time() - t0:.0f}s", flush=True)

    images = sorted(gt["images"], key=lambda im: im["file_name"])
    ordered_ids = [im["image_id"] for im in images]
    n_total = len(ordered_ids)
    gt_obj, gt_dropped = clean(gt["annotations"])
    pr_obj, pr_dropped = clean(pred["predictions"])
    frames = sorted(set(min(n, n_total) for n in a.frames) | {n_total})

    result = {"seq": seq, "n_frames_total": n_total, "gt_dropped_non_finite": gt_dropped,
              "pred_dropped_non_finite": pr_dropped, "prefixes": {}}
    for n in frames:
        keep = set(ordered_ids[:n])
        g = dict(gt)
        g["images"] = [im for im in images if im["image_id"] in keep]
        g["annotations"] = [e for e in gt_obj if e.get("image_id") in keep]
        p = dict(pred)
        p["predictions"] = [e for e in pr_obj if e.get("image_id") in keep]
        t1 = time.time()
        rows = score(seq, g, p)
        rows["n_frames"] = n
        rows["n_pred_entries"] = len(p["predictions"])
        result["prefixes"][str(n)] = rows
        on, off = rows["attributes ON (official GS-HOTA)"], rows["attributes OFF (geometry + association)"]
        print(f"  {n:>6} frames: official {on['HOTA']:6.2f} (DetA {on['DetA']:5.2f} AssA {on['AssA']:5.2f}"
              f" LocA {on['LocA']:5.2f})  attrs off {off['HOTA']:6.2f}  ids {on['ids']}/{on['gt_ids']}"
              f"  [{time.time() - t1:.0f}s]", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

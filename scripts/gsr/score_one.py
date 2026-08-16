"""Score one SoccerNetGS prediction file with GS-HOTA, tolerating unlocalisable detections.

TrackLab's built-in evaluator dies with `ValueError: matrix contains invalid numeric entries`
when any prediction carries a NaN or infinite pitch coordinate, which the calibration produces
for detections near the horizon (the homography's w goes to zero there). On the 10-minute run
that was 470 of 298,896 predictions -- 0.16% -- and it cost the whole score.

A detection with no finite pitch position cannot be matched against anything: GS-HOTA is
computed in pitch space. Dropping it is exactly equivalent to the tracker not having reported
it, which is the honest reading, and it is what the evaluator should have done instead of
throwing. The count is always printed, so the reader can see how much was discarded.

    python scripts/gsr/score_one.py --pred <pred.json> --gt <Labels-GameState.json>
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--seq", default=None, help="sequence name (default: gt's parent dir)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    seq = a.seq or a.gt.parent.name
    pred = json.loads(a.pred.read_text())
    entries = pred["predictions"]

    kept, dropped = [], 0
    for e in entries:
        if e.get("supercategory") == "object" and not finite_pitch(e):
            dropped += 1
            continue
        kept.append(e)
    n_obj = sum(1 for e in entries if e.get("supercategory") == "object")
    print(f"{seq}: {n_obj:,} object predictions, dropped {dropped:,} "
          f"({100.0 * dropped / max(1, n_obj):.2f}%) with non-finite pitch coordinates")
    pred["predictions"] = kept

    # trackeval wants a directory tree, so lay one out in a temp dir
    tmp = Path(tempfile.mkdtemp(prefix="gshota_"))
    try:
        # SKIP_SPLIT_FOL=True, so GT sits directly under GT_FOLDER with no split directory
        gt_root = tmp / "gt" / seq
        gt_root.mkdir(parents=True)
        shutil.copy(a.gt, gt_root / "Labels-GameState.json")
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
        print()
        for label, r in rows.items():
            print(f"  {label}")
            print(f"      GS-HOTA {r['HOTA']:6.3f}   DetA {r['DetA']:6.3f}   "
                  f"AssA {r['AssA']:6.3f}   LocA {r['LocA']:6.3f}")
        any_row = next(iter(rows.values()))
        print(f"\n  {any_row['ids']} predicted tracklets vs {any_row['gt_ids']} real players")
        if a.out:
            a.out.write_text(json.dumps(
                {"seq": seq, "scores": rows,
                 "dropped_non_finite": dropped, "n_object_predictions": n_obj}, indent=2))
            print(f"  wrote {a.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

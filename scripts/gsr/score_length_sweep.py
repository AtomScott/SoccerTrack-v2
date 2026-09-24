"""Score the sequence-length sweep two ways, so length is separated from content.

FULL WINDOW   each run scored on its own whole length -> "what do you get on N minutes"
COMMON WINDOW every run scored on the SAME first 750 frames -> isolates the effect of
              length itself, because the footage being compared is identical

The common window is the scientifically clean reading. Most of the pipeline is causal and
cannot be changed by later frames, but GTALink, MajorityVoteTracklet, TrackletTeamClustering
and TrackletTeamSideLabeling are global over the whole video, so a longer run can change the
result on the very same opening 30 seconds. That is the effect being measured.

Each prefix sequence carries its own image_id prefix (seq ids 921..925), so id strings differ
between runs. That is fine: within a run GT and predictions agree, and the underlying footage
is the same, so the scores are comparable.

Attributes are scored both ON (official GS-HOTA) and OFF (geometry + association only), since
at 30 s the attributes were already costing 35 of 64 available points.
"""
from __future__ import annotations
import argparse, json, shutil, sys, tempfile
from pathlib import Path

import numpy as np
import trackeval as te
from trackeval.datasets import SoccerNetGS

_ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--sequences", default="/mnt/storage/SoccerTrack-v2/SoccerNetGS/test",
                 help="dir holding the staged prefix sequences")
_ap.add_argument("--outputs", default="/home/atom/soccernet/gsr/outputs",
                 help="TrackLab outputs root, searched for each run's predictions")
_ap.add_argument("--common-frames", type=int, default=750,
                 help="length of the shared window every run is also scored on")
_ap.add_argument("--out", default="sweep_scores.json")
_args = _ap.parse_args()
ROOT = Path(_args.sequences)
OUTS = Path(_args.outputs)
COMMON_FRAMES = _args.common_frames


def _score(gt_root: Path, pred_root: Path, seq: str, attrs: bool) -> dict:
    ec = te.Evaluator.get_default_eval_config()
    ec.update({"PRINT_CONFIG": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
               "PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
               "PLOT_CURVES": False, "USE_PARALLEL": False})
    dc = SoccerNetGS.get_default_dataset_config()
    dc.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(pred_root),
               "TRACKERS_TO_EVAL": ["t"], "TRACKER_SUB_FOLDER": "data",
               "SKIP_SPLIT_FOL": True, "SEQ_INFO": {seq: None}, "EVAL_SPACE": "pitch",
               "OUTPUT_FOLDER": None, "PRINT_CONFIG": False,
               "USE_ROLES": attrs, "USE_TEAMS": attrs, "USE_JERSEY_NUMBERS": attrs})
    res, _ = te.Evaluator(ec).evaluate([SoccerNetGS(dc)], [te.metrics.HOTA()])
    h = res["SoccerNetGS"]["t"][seq]["person"]["HOTA"]
    c = res["SoccerNetGS"]["t"][seq]["person"]["Count"]
    g = lambda k: float(np.mean(np.atleast_1d(h[k]))) * 100
    out = {k: g(k) for k in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr")}
    out.update({"dets": int(c["Dets"]), "gt_dets": int(c["GT_Dets"]),
                "ids": int(c["IDs"]), "gt_ids": int(c["GT_IDs"]), "frames": int(c["Frames"])})
    return out


def _stage(gt_labels: Path, pred_json: Path, seq: str, limit: int | None):
    """Build a throwaway GT+pred pair, optionally truncated to the first `limit` frames."""
    tmp = Path(tempfile.mkdtemp(prefix="sweep-score-"))
    gt = json.loads(gt_labels.read_text())
    imgs = sorted(gt["images"], key=lambda im: int(str(im["image_id"]).split("_")[-1]))
    if limit is not None:
        imgs = imgs[:limit]
    keep = {im["image_id"] for im in imgs}
    gt_out = {"info": dict(gt["info"]), "images": imgs,
              "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
              "categories": gt["categories"]}
    gt_out["info"]["seq_length"] = len(imgs)
    d = tmp / "gt" / seq
    d.mkdir(parents=True)
    (d / "Labels-GameState.json").write_text(json.dumps(gt_out))

    pred = json.loads(pred_json.read_text())["predictions"]
    pd_dir = tmp / "pred" / "t" / "data"
    pd_dir.mkdir(parents=True)
    (pd_dir / f"{seq}.json").write_text(json.dumps(
        {"predictions": [p for p in pred if p.get("image_id") in keep]}))
    return tmp


def find_pred(exp: str, seq: str) -> Path | None:
    cands = sorted(OUTS.glob(f"{exp}/*/*/eval/pred/SoccerNetGS-test/t*/{seq}.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        cands = sorted(OUTS.glob(f"{exp}/*/*/eval/pred/**/{seq}.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


SPECS = [("30s", 750), ("1min", 1500), ("5min", 7500), ("15min", 22500), ("30min", 45000)]

rows = []
for label, frames in SPECS:
    seq = f"CLPD-128057-1st-{label}"
    pred = find_pred(f"sweep-{label}", seq)
    if pred is None:
        print(f"{label:6} no predictions yet, skipping", flush=True)
        continue
    gt_labels = ROOT / seq / "Labels-GameState.json"
    for window, limit in (("full", None), ("common750", COMMON_FRAMES)):
        tmp = _stage(gt_labels, pred, seq, limit)
        try:
            for attrs in (True, False):
                m = _score(tmp / "gt", tmp / "pred", seq, attrs)
                m.update({"label": label, "frames": frames, "window": window,
                          "attrs": "on" if attrs else "off"})
                rows.append(m)
                print(f"{label:6} {window:10} attrs={'on ' if attrs else 'off'} "
                      f"HOTA={m['HOTA']:6.2f} DetA={m['DetA']:6.2f} AssA={m['AssA']:6.2f} "
                      f"LocA={m['LocA']:6.2f} ids={m['ids']}/{m['gt_ids']}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

out = Path(_args.out)
out.write_text(json.dumps(rows, indent=2))
print(f"\nwrote {out}  ({len(rows)} rows)")

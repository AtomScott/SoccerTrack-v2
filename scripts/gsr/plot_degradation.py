"""Chart how GSR accuracy degrades with sequence length, and show why.

Three panels, because the headline number alone does not explain itself:

  1. GS-HOTA vs sequence length -- what actually happens to the score
  2. the attributes that drive it -- team and jersey accuracy, measured on
     geometry-matched pairs, since GS-HOTA collapses when either is wrong
  3. the mechanism -- predicted tracklet count against the 22 real players; every
     attribute failure so far traces back to fragmentation

All points must come from ONE configuration. Mixing configs across lengths would attribute a
config change to sequence length, which is exactly the confound this chart exists to avoid.

    python scripts/gsr/plot_degradation.py --runs deg --out degradation.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

FPS = 25
MATCH_TOL_M = 2.0   # only count confident geometric matches when scoring attributes


def attribute_accuracy(gt_path: Path, pred_path: Path) -> dict:
    """Team/jersey accuracy on geometry-matched pairs, plus predicted tracklet count."""
    gt = json.loads(gt_path.read_text())
    G = defaultdict(list)
    gt_tracks = set()
    for a in gt["annotations"]:
        if a.get("supercategory") != "object":
            continue
        bp, at = a["bbox_pitch"], a.get("attributes", {})
        G[a["image_id"]].append((bp["x_bottom_middle"], bp["y_bottom_middle"],
                                 at.get("team"), str(at.get("jersey"))))
        gt_tracks.add(a["track_id"])
    del gt

    preds = json.loads(pred_path.read_text())["predictions"]
    P = defaultdict(list)
    tids = set()
    for p in preds:
        if p.get("supercategory") != "object":
            continue
        bp, at = p.get("bbox_pitch") or {}, p.get("attributes", {})
        if "x_bottom_middle" not in bp:
            continue
        P[p["image_id"]].append((bp["x_bottom_middle"], bp["y_bottom_middle"],
                                 at.get("team"), str(at.get("jersey"))))
        tids.add(p.get("track_id"))
    del preds

    tm = jn = n = 0
    for iid, gs in G.items():
        ps = P.get(iid)
        if not ps:
            continue
        A = np.array([[g[0], g[1]] for g in gs])
        B = np.array([[q[0], q[1]] for q in ps])
        C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > MATCH_TOL_M:
                continue
            n += 1
            tm += (gs[i][2] == ps[j][2])
            jn += (gs[i][3] == ps[j][3])
    return {"team": 100.0 * tm / n if n else float("nan"),
            "jersey": 100.0 * jn / n if n else float("nan"),
            "tracklets": len(tids), "gt_tracks": len(gt_tracks), "matched": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="degradation.csv (label,frames,gs_hota,...)")
    ap.add_argument("--sequences", default="/mnt/storage/SoccerTrack-v2/SoccerNetGS/test")
    ap.add_argument("--outputs", default="/home/atom/soccernet/gsr/outputs")
    ap.add_argument("--exp-prefix", default="deg", help="experiment_name prefix of the runs")
    ap.add_argument("--source", default="CLPD-128057-1st")
    ap.add_argument("--out", default="degradation.png")
    ap.add_argument("--title", default="SoccerTrack v2 GSR: accuracy vs sequence length")
    ap.add_argument("--compare-csv", default=None,
                    help="optional second curve (label,frames,gs_hota) to overlay on panel 1")
    ap.add_argument("--compare-label", default="alternative config")
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(a.csv)):
        if r.get("gs_hota") in ("NA", "", None):
            continue
        lab = r["label"]
        seq = f"{a.source}-{lab}"
        gt = Path(a.sequences) / seq / "Labels-GameState.json"
        cands = sorted(Path(a.outputs).glob(f"{a.exp_prefix}-{lab}/*/*/eval/pred/**/{seq}.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        acc = attribute_accuracy(gt, cands[0]) if (cands and gt.exists()) else {}
        rows.append({"label": lab, "frames": int(r["frames"]),
                     "minutes": int(r["frames"]) / FPS / 60.0,
                     "gs_hota": float(r["gs_hota"]), **acc})
        print(f"  {lab:6} {rows[-1]['minutes']:5.1f} min  GS-HOTA {rows[-1]['gs_hota']:6.2f}"
              f"  team {acc.get('team', float('nan')):5.1f}%  jersey {acc.get('jersey', float('nan')):5.1f}%"
              f"  tracklets {acc.get('tracklets', 0)}", flush=True)
    if not rows:
        print("no usable rows"); return 1
    rows.sort(key=lambda r: r["minutes"])
    x = [r["minutes"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(a.title + "\nmatch 128057 1st half, nested prefixes of identical footage, "
                 "single configuration (jersey gates 100/0.6 + detection-level team)",
                 fontsize=11)

    # 1 -- the score
    ax[0].plot(x, [r["gs_hota"] for r in rows], "-o", color="#b3261e", lw=2, ms=7,
               label="detection-level team (stable)")
    for r in rows:
        ax[0].annotate(f"{r['gs_hota']:.1f}", (r["minutes"], r["gs_hota"]),
                       textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8,
                       color="#b3261e")
    if a.compare_csv and Path(a.compare_csv).exists():
        c = [r for r in csv.DictReader(open(a.compare_csv)) if r.get("gs_hota") not in ("NA", "", None)]
        c.sort(key=lambda r: int(r["frames"]))
        cx = [int(r["frames"]) / FPS / 60.0 for r in c]
        cy = [float(r["gs_hota"]) for r in c]
        ax[0].plot(cx, cy, "--s", color="#0b5fa5", lw=2, ms=6, label=a.compare_label)
        for xx, yy in zip(cx, cy):
            ax[0].annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8, color="#0b5fa5")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("sequence length (minutes, log)")
    ax[0].set_ylabel("GS-HOTA (%)")
    ax[0].set_title("1. GS-HOTA peaks at 1 min, then falls\n"
                    "and the best team module depends on length", fontsize=10)
    ax[0].grid(alpha=.3)
    ax[0].set_ylim(bottom=0)
    ax[0].legend(fontsize=8, loc="lower left")

    # 2 -- the attributes responsible
    if any("team" in r for r in rows):
        ax[1].plot(x, [r.get("team", np.nan) for r in rows], "-o", color="#0f7a4a",
                   lw=2, ms=6, label="team accuracy")
        ax[1].plot(x, [r.get("jersey", np.nan) for r in rows], "-s", color="#0b5fa5",
                   lw=2, ms=6, label="jersey accuracy")
        ax[1].axhline(50, ls=":", color="#b3261e", lw=1.2,
                      label="chance for team (2 classes)")
        ax[1].set_ylim(0, 100)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("sequence length (minutes, log)")
    ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("2. Jersey drives the decline; team holds\n"
                    "(detection-level team: stable but never excellent)", fontsize=10)
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8, loc="lower left")

    # 3 -- the mechanism
    if any("tracklets" in r for r in rows):
        ax[2].plot(x, [r.get("tracklets", np.nan) for r in rows], "-o",
                   color="#8e44ad", lw=2, ms=6, label="predicted tracklets")
        gtt = next((r["gt_tracks"] for r in rows if r.get("gt_tracks")), 22)
        ax[2].axhline(gtt, ls="--", color="k", lw=1.2,
                      label=f"real players ({gtt})")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("sequence length (minutes, log)")
    ax[2].set_ylabel("tracklets")
    ax[2].set_title("3. Mechanism: fragmentation grows ~linearly\n"
                    "spreading jersey reads across more tracklets", fontsize=10)
    ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=8, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = Path(a.out)
    plt.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Chart GSR accuracy against sequence length, and show what actually breaks.

Three panels, because the headline number does not explain itself:
  1. GS-HOTA vs length, official metric and attributes-off, on identical footage
  2. where the points go -- attribute accuracy measured on geometry-matched pairs
  3. the tracklet story, separating real fragmentation from a tail of junk tracklets

    python scripts/gsr/plot_length_curve.py --csv results/gsr/length_sweep_128057_1st.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FPS = 25
# measured on geometry-matched pairs (<=2 m), 1 fps sample -- see scripts/gsr/diagnose_length.py
ATTRS = {   # minutes: (role %, team %, jersey %)
    5.0:  (100.0, 86.6, 31.0),
    45.0: (97.92, 76.17, 45.95),
}
# tracklets: total, and those holding >=100 detections ("substantial")
TRACKLETS = {0.5: (26, None), 1.0: (35, None), 2.0: (42, None),
             5.0: (60, None), 10.0: (99, None), 45.0: (347, 65)}
ATTRS_OFF = {10.0: 33.908, 45.0: 26.036}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/gsr/length_sweep_128057_1st.csv")
    ap.add_argument("--compare", default="results/gsr/length_sweep_128057_1st_tracklet_team.csv")
    ap.add_argument("--out", default="results/gsr/length_curve.png")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if r.get("gs_hota") not in ("", "NA", None)]
    rows.sort(key=lambda r: int(r["frames"]))
    x = [int(r["frames"]) / FPS / 60 for r in rows]
    y = [float(r["gs_hota"]) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle("SoccerTrack v2 GSR baseline: accuracy collapses with sequence length\n"
                 "match 128057 1st half, nested prefixes of identical footage, one configuration "
                 "(jersey gates 100/0.6 + detection-level team)", fontsize=11)

    # 1 -- the score
    ax[0].plot(x, y, "-o", color="#b3261e", lw=2.2, ms=7, label="official GS-HOTA", zorder=3)
    for xx, yy in zip(x, y):
        ax[0].annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points",
                       xytext=(0, -15), ha="center", fontsize=8.5, color="#b3261e")
    ox = sorted(ATTRS_OFF); ax[0].plot(ox, [ATTRS_OFF[k] for k in ox], "--s",
                                       color="#0b5fa5", lw=1.8, ms=6,
                                       label="attributes off (geometry + association)")
    if Path(a.compare).exists():
        c = [r for r in csv.DictReader(open(a.compare)) if r.get("gs_hota") not in ("", "NA", None)]
        c.sort(key=lambda r: int(r["frames"]))
        ax[0].plot([int(r["frames"]) / FPS / 60 for r in c], [float(r["gs_hota"]) for r in c],
                   ":^", color="#0f7a4a", lw=1.6, ms=6, label="stock tracklet-level team")
    ax[0].axvline(1.0, color="#999", ls=":", lw=1)
    ax[0].annotate("peak", (1.0, max(y)), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("sequence length (minutes, log)")
    ax[0].set_ylabel("GS-HOTA (%)"); ax[0].set_ylim(bottom=0)
    ax[0].set_title("1. Peaks at 1 min, then falls monotonically\n"
                    "45 min is 38% of the 1 min score", fontsize=10)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, loc="lower left")

    # 2 -- where the points go
    ms = sorted(ATTRS)
    for i, (lab, col, mk) in enumerate((("role", "#0f7a4a", "o"),
                                        ("team", "#0b5fa5", "s"),
                                        ("jersey", "#b3261e", "^"))):
        ax[1].plot(ms, [ATTRS[m][i] for m in ms], f"-{mk}", color=col, lw=2, ms=7, label=lab)
        for m in ms:
            ax[1].annotate(f"{ATTRS[m][i]:.0f}", (m, ATTRS[m][i]), textcoords="offset points",
                           xytext=(0, 6), ha="center", fontsize=8, color=col)
    ax[1].axhline(50, ls=":", color="#666", lw=1.2, label="chance for team (2 classes)")
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("sequence length (minutes, log)"); ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("2. Role is solved. Jersey is the hole.\n"
                    "A wrong attribute is no match at all, not partial credit", fontsize=10)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, loc="lower left")

    # 3 -- the tracklet story
    tm = sorted(TRACKLETS)
    ax[2].plot(tm, [TRACKLETS[m][0] for m in tm], "-o", color="#8e44ad", lw=2, ms=7,
               label="predicted tracklets (all)")
    sub = [(m, TRACKLETS[m][1]) for m in tm if TRACKLETS[m][1]]
    if sub:
        ax[2].plot([s[0] for s in sub], [s[1] for s in sub], "D", color="#8e44ad",
                   ms=9, mfc="white", mew=2, label="holding ≥100 detections")
        ax[2].annotate(f"{sub[-1][1]}", sub[-1], textcoords="offset points",
                       xytext=(8, -2), fontsize=9, color="#8e44ad")
    ax[2].annotate(f"{TRACKLETS[45.0][0]}", (45.0, TRACKLETS[45.0][0]),
                   textcoords="offset points", xytext=(-4, 8), ha="right", fontsize=9,
                   color="#8e44ad")
    ax[2].axhline(23, ls="--", color="k", lw=1.3, label="real players (23)")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("sequence length (minutes, log)"); ax[2].set_ylabel("tracklets (log)")
    ax[2].set_title("3. 347 tracklets, but 282 of them are junk\n"
                    "holding 0.2% of detections; ~65 carry the rest", fontsize=10)
    ax[2].grid(alpha=.3, which="both"); ax[2].legend(fontsize=8, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

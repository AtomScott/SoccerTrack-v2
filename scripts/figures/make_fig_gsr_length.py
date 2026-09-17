"""fig5_gsr_length.pdf (Fig. 4 of the paper): mean GS-HOTA over the twenty
halves against the length scored, one line chart.

Filled markers: the whole-half predictions of every half (Table
tab:gsr_results) rescored over their first 30 s, 1, 2, 5 and 10 minutes and
the whole half (scripts/gsr/score_prefixes.py, compiled by
compile_prefix_rescoring.py into results/gsr/prefix_rescoring_all_halves.csv),
mean over the twenty halves, official metric (black) and attributes off
(blue); bands span the twenty halves. Hollow markers: the mean of the
independent 30 s runs (results/gsr/sweep30s_all_matches.csv), i.e. the 30 s
column of tab:gsr_results. The 20-minute window is in the supplementary table
and left out of the figure.

Style matches make_fig3_v2.py (Okabe-Ito, DejaVu Sans 7 pt, 372 pt width).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/home/atom/SoccerTrack-v2"
OUT_DIR = "/home/atom/soccertrack-v2/paper/figures"

BLUE = "#0072B2"
INK = "#1a1a1a"
MUTED = "#555555"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.labelsize": 7,
    "axes.labelcolor": INK,
    "text.color": INK,
})

LABELS = ["30s", "1min", "2min", "5min", "10min", "full"]
XPOS = {"30s": 0.5, "1min": 1, "2min": 2, "5min": 5, "10min": 10, "full": 46.6}
TICK = {"30s": "30 s", "1min": "1 min", "2min": "2 min", "5min": "5 min",
        "10min": "10 min", "full": "whole half\n(45 to 49 min)"}

rows = list(csv.DictReader(open(os.path.join(REPO, "results/gsr/prefix_rescoring_all_halves.csv"))))
sweep = list(csv.DictReader(open(os.path.join(REPO, "results/gsr/sweep30s_all_matches.csv"))))
assert len(sweep) == 20


def stats(key):
    means, lo, hi = [], [], []
    for label in LABELS:
        v = [float(r[key]) for r in rows if r["label"] == label]
        assert len(v) == 20, (label, len(v))
        means.append(sum(v) / 20)
        lo.append(min(v))
        hi.append(max(v))
    return means, lo, hi


ind_on = sum(float(r["gshota_official"]) for r in sweep) / 20
ind_off = sum(float(r["gshota_attrs_off"]) for r in sweep) / 20

x = [XPOS[l] for l in LABELS]
fig, ax = plt.subplots(figsize=(5.15, 2.7))
for key, color in (("off_hota", BLUE), ("hota", INK)):
    means, lo, hi = stats(key)
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(x, means, color=color, linewidth=1.8, marker="o", markersize=4, zorder=4)
    for xi, mi in zip(x, means):
        ax.annotate(f"{mi:.1f}", (xi, mi), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.2, color=color)
for val, color in ((ind_on, INK), (ind_off, BLUE)):
    ax.scatter([0.5], [val], s=26, facecolor="white", edgecolor=color, linewidth=1.0, zorder=5)
    ax.annotate(f"{val:.1f}", (0.5, val), textcoords="offset points", xytext=(-7, 0),
                ha="right", va="center", fontsize=6.2, color=color)

handles = [
    Line2D([], [], color=BLUE, linewidth=1.8, marker="o", markersize=4,
           label="attributes off, mean over the twenty halves"),
    Line2D([], [], color=INK, linewidth=1.8, marker="o", markersize=4,
           label="official GS-HOTA, mean over the twenty halves"),
    Line2D([], [], color=INK, linestyle="none", marker="o", markersize=4.5,
           markerfacecolor="white", label="independent 30 s runs, mean"),
]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.2)
ax.text(0.53, 3, "bands: range over the twenty halves", fontsize=6.0, color=MUTED, ha="left")
ax.set_xscale("log")
ax.set_xticks(x)
ax.set_xticklabels([TICK[l] for l in LABELS])
ax.minorticks_off()
ax.set_xlim(0.3, 62)
ax.set_ylim(0, 80)
ax.set_xlabel("length scored from kickoff (log scale)")
ax.set_ylabel("GS-HOTA")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.3)
out = os.path.join(OUT_DIR, "fig5_gsr_length.pdf")
fig.savefig(out)
print("wrote", out, f"| means official {[round(m, 2) for m in stats('hota')[0]]}"
      f" attrs-off {[round(m, 2) for m in stats('off_hota')[0]]} | independent 30 s {ind_on:.2f} / {ind_off:.2f}")

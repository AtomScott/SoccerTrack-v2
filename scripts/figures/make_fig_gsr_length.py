"""fig5_gsr_length.pdf (Fig. 4 of the paper): GS-HOTA against sequence length,
one line chart.

Thin grey lines: each of the twenty halves, from its opening 30 s to its
whole half (44.9 to 49.0 min at 25 fps), official GS-HOTA, the values of
tab:gsr_results. Black: the mean over the twenty halves at both lengths
(29.92 at 30 s, 11.79 over the whole half, plotted at the mean half length).
Blue: nested prefixes of 128057, first half (Supplementary Table
tab:gsr_length), the only half scored at intermediate lengths.

Data: results/gsr/sweep30s_all_matches.csv (30 s), results/gsr/
full_table_all_matches.csv (chosen rows, whole half), per-half frame counts
from the cloud summary.txt files (results/gsr/cloud/<fleet>/CLPD-*/) and the
staged Labels-GameState.json of the three workstation halves.

Style matches make_fig3_v2.py (Okabe-Ito, DejaVu Sans 7 pt, 372 pt width).
"""
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/atom/SoccerTrack-v2"
OUT_DIR = "/home/atom/soccertrack-v2/paper/figures"
FPS = 25.0

BLUE = "#0072B2"
RED = "#D55E00"
INK = "#1a1a1a"
MUTED = "#555555"
HALF_LINE = "#9a9a9a"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "axes.labelsize": 7,
    "axes.labelcolor": INK,
    "text.color": INK,
})

LOCAL_GT = {
    ("117092", "1st"): "/mnt/storage/SoccerTrack-v2/SoccerNetGS-117092/test/CLPD-117092-1st/Labels-GameState.json",
    ("128057", "1st"): "/mnt/storage/SoccerTrack-v2/SoccerNetGS-128057/test/CLPD-128057-1st/Labels-GameState.json",
    ("128057", "2nd"): "/mnt/storage/SoccerTrack-v2/SoccerNetGS-128057/test/CLPD-128057-2nd/Labels-GameState.json",
}


def load_halves():
    sweep = {}
    with open(os.path.join(REPO, "results/gsr/sweep30s_all_matches.csv")) as f:
        for r in csv.DictReader(f):
            sweep[(r["match"], r["half"])] = float(r["gshota_official"])
    full, frames = {}, {}
    with open(os.path.join(REPO, "results/gsr/full_table_all_matches.csv")) as f:
        for r in csv.DictReader(f):
            if r["chosen"] != "True":
                continue
            k = (r["match"], r["half"])
            full[k] = float(r["hota"])
            if r["source"] == "local":
                with open(LOCAL_GT[k]) as g:
                    frames[k] = len(json.load(g)["images"])
            else:
                p = r["path"].replace("/mnt/storage/SoccerTrack-v2/gsr-cloud-results/",
                                      os.path.join(REPO, "results/gsr/cloud/"))
                summ = open(os.path.join(os.path.dirname(p), "summary.txt")).read()
                frames[k] = int(re.search(r"frames=(\d+)", summ).group(1))
    keys = sorted(full)
    assert len(keys) == 20 and all(k in sweep for k in keys), "need all twenty halves"
    return [(k, sweep[k], full[k], frames[k] / FPS / 60.0) for k in keys]


halves = load_halves()
mean30 = sum(h[1] for h in halves) / 20
meanfull = sum(h[2] for h in halves) / 20
meanlen = sum(h[3] for h in halves) / 20
assert all(h[2] < h[1] for h in halves), "every half must fall"

# Supplementary Table tab:gsr_length: length (min), GS-HOTA. The 30 s value is
# the raw scorer output 37.1549 rounded once (37.15) so that the figure agrees
# with tab:gsr_results; the length-sweep CSV stores 37.155.
prefix_len = [0.5, 1, 2, 5, 10, 67625 / FPS / 60.0]
prefix_gs = [37.15, 47.59, 39.88, 30.97, 25.60, 18.09]

fig, ax = plt.subplots(figsize=(5.15, 2.7))

ax.axvline(0.5, color=RED, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
ax.text(0.53, 59.5, "existing GSR benchmark clips (30 s)", color=RED,
        fontsize=6.2, ha="left", va="top")

for (m, h), s30, sf, ln in halves:
    ax.plot([0.5, ln], [s30, sf], color=HALF_LINE, linewidth=0.6, alpha=0.8,
            zorder=2, solid_capstyle="round")

ax.plot([0.5, meanlen], [mean30, meanfull], color=INK, linewidth=1.8,
        marker="o", markersize=3.6, zorder=4, clip_on=False)
ax.annotate(f"{mean30:.2f}", (0.5, mean30), textcoords="offset points",
            xytext=(-6, 0), ha="right", va="center", fontsize=6.2, color=INK,
            fontweight="bold")
ax.annotate(f"{meanfull:.2f}", (meanlen, meanfull), textcoords="offset points",
            xytext=(7, 0), ha="left", va="center", fontsize=6.2, color=INK,
            fontweight="bold")

ax.plot(prefix_len, prefix_gs, color=BLUE, linewidth=1.2, marker="o",
        markersize=3.2, zorder=5, clip_on=False)
for x, y in zip(prefix_len, prefix_gs):
    if x == 1:
        ax.annotate(f"{y:.2f} (peak)", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color=BLUE,
                    fontweight="bold")
    elif x == 0.5:
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(-6, 0), ha="right", va="center", fontsize=6.2, color=BLUE)
    elif x > 40:
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(7, 0), ha="left", va="center", fontsize=6.2, color=BLUE)
    else:
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color=BLUE)

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], color=BLUE, linewidth=1.2, marker="o", markersize=3.2,
           label="128057, first half, nested prefixes"),
    Line2D([], [], color=INK, linewidth=1.8, marker="o", markersize=3.6,
           label="mean of the twenty halves"),
    Line2D([], [], color=HALF_LINE, linewidth=0.8,
           label="one line per half, 30 s to whole half"),
]
ax.legend(handles=handles, loc="upper right", fontsize=6.2, frameon=False,
          handlelength=2.2, handletextpad=0.6, borderaxespad=0.4, labelspacing=0.5)

ax.set_xscale("log")
ticks = [0.5, 1, 2, 5, 10, 45]
ax.set_xticks(ticks)
ax.set_xticklabels(["30 s", "1 min", "2 min", "5 min", "10 min", "whole half\n(45 to 49 min)"])
ax.minorticks_off()
ax.set_xlim(0.42, 62)
ax.set_ylim(0, 60)
ax.set_xlabel("sequence length (log scale)")
ax.set_ylabel("GS-HOTA (official)")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.3)
out = os.path.join(OUT_DIR, "fig5_gsr_length.pdf")
fig.savefig(out)
print("wrote", out, f"mean30={mean30:.2f} meanfull={meanfull:.2f} meanlen={meanlen:.1f} min")

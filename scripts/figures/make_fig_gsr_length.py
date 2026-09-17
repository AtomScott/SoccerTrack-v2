"""fig5_gsr_length.pdf (Fig. 4 of the paper): two panels on sequence length.

(a) Every half of the corpus at two lengths: GS-HOTA on the opening 30 s
    (x) against GS-HOTA over the whole half (y), official (filled) and with
    attribute matching disabled (hollow). All forty points lie below the
    identity line; the guide lines mark a factor of two and four. Halves of
    the three weakest matches (117092, 132831, 132877) are drawn in
    vermilion, the other seven matches in blue; the released test split
    (128057, 132831) uses square markers.
    Data: results/gsr/sweep30s_all_matches.csv (30 s) and
    results/gsr/full_table_all_matches.csv (chosen rows, full half), i.e.
    the same numbers as tab:gsr_results.
(b) GS-HOTA against sequence length on nested prefixes of 128057, first
    half (Supplementary Table tab:gsr_length), unchanged from the earlier
    single-panel figure.

Style matches make_fig3_v2.py (Okabe-Ito, DejaVu Sans 7 pt, 372 pt width).
Palette (#0072B2, #D55E00) validated with the dataviz skill's checker.
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
RED = "#D55E00"
INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#c8c8c8"

WEAK = {"117092", "132831", "132877"}
TEST = {"128057", "132831"}

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


def load_pairs():
    sweep = {}
    with open(os.path.join(REPO, "results/gsr/sweep30s_all_matches.csv")) as f:
        for r in csv.DictReader(f):
            sweep[(r["match"], r["half"])] = (float(r["gshota_official"]),
                                              float(r["gshota_attrs_off"]))
    full = {}
    with open(os.path.join(REPO, "results/gsr/full_table_all_matches.csv")) as f:
        for r in csv.DictReader(f):
            if r["chosen"] == "True":
                full[(r["match"], r["half"])] = (float(r["hota"]), float(r["off_hota"]))
    keys = sorted(full)
    assert len(keys) == 20 and all(k in sweep for k in keys), "need all twenty halves"
    return [(k, sweep[k], full[k]) for k in keys]


fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.35),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

# ---------------------------------------------------------------- (a)
pairs = load_pairs()
lim = 82
ax.plot([0, lim], [0, lim], color=MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
ax.text(lim - 1, lim - 4, "equal", color=MUTED, fontsize=6.0, ha="right", va="top")
for factor, va, dy in ((2, "bottom", 0.8), (4, "top", -0.8)):
    ax.plot([0, lim], [0, lim / factor], color=GRID, linewidth=0.6, zorder=1)
    ax.text(lim - 1, lim / factor + dy, f"1/{factor}", color=MUTED, fontsize=6.0,
            ha="right", va=va)

for (m, h), (s30, s30off), (sf, sfoff) in pairs:
    color = RED if m in WEAK else BLUE
    marker = "s" if m in TEST else "o"
    ax.scatter([s30], [sf], s=16, marker=marker, facecolor=color, edgecolor=color,
               linewidth=0.6, zorder=3)
    ax.scatter([s30off], [sfoff], s=16, marker=marker, facecolor="white",
               edgecolor=color, linewidth=0.8, zorder=3)

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal")
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_xlabel("GS-HOTA, opening 30 s")
ax.set_ylabel("GS-HOTA, whole half")
ax.spines[["top", "right"]].set_visible(False)

handles = [
    Line2D([], [], marker="o", color=BLUE, linestyle="none", markersize=4,
           label="seven other matches"),
    Line2D([], [], marker="o", color=RED, linestyle="none", markersize=4,
           label="117092, 132831, 132877"),
    Line2D([], [], marker="s", color=INK, linestyle="none", markersize=4,
           markerfacecolor="none", label="test split"),
    Line2D([], [], marker="o", color=INK, linestyle="none", markersize=4,
           label="official"),
    Line2D([], [], marker="o", color=INK, linestyle="none", markersize=4,
           markerfacecolor="white", label="attributes off"),
]
ax.legend(handles=handles, loc="upper left", fontsize=5.8, frameon=False,
          handletextpad=0.3, borderaxespad=0.2, labelspacing=0.35)
ax.text(-0.22, 1.02, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

# ---------------------------------------------------------------- (b)
# Table tab:gsr_length: length (min), GS-HOTA. The 30 s value is the raw
# scorer output 37.1549 (score_30s_128057_1st.json) rounded once, 37.15, so
# that the figure agrees with tab:gsr_results and tab:gsr_length; the
# length-sweep CSV stores 37.155, which re-rounds to 37.16.
lengths = [0.5, 1, 2, 5, 10, 45]
gshota = [37.15, 47.59, 39.88, 30.97, 25.60, 18.09]
tick_labels = ["30 s", "1 min", "2 min", "5 min", "10 min", "45 min\n(whole half)"]

bx.axvline(0.5, color=RED, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
bx.text(0.53, 4.0, "existing GSR\nbenchmark clips (30 s)", color=RED,
        fontsize=6.0, ha="left", va="bottom")
bx.plot(lengths, gshota, color=BLUE, linewidth=1.2, marker="o",
        markersize=3.2, zorder=3, clip_on=False)
for x, y in zip(lengths, gshota):
    if x == 1:
        bx.annotate(f"{y:.2f} (peak)", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color=INK,
                    fontweight="bold")
    elif x == 45:
        bx.annotate(f"{y:.2f}\n(38% of peak)", (x, y),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=6.2, color=INK)
    elif x == 0.5:
        bx.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(7, -9), ha="left", fontsize=6.2, color=MUTED)
    else:
        bx.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color=MUTED)
bx.set_xscale("log")
bx.set_xticks(lengths)
bx.set_xticklabels(tick_labels)
bx.minorticks_off()
bx.set_xlim(0.42, 54)
bx.set_ylim(0, 55)
bx.set_xlabel("nested prefixes of 128057, first half (log scale)")
bx.set_ylabel("GS-HOTA")
bx.spines[["top", "right"]].set_visible(False)
bx.text(-0.16, 1.02, "b", transform=bx.transAxes, fontsize=8, fontweight="bold")

fig.tight_layout(pad=0.3, w_pad=1.2)
out = os.path.join(OUT_DIR, "fig5_gsr_length.pdf")
fig.savefig(out)
print("wrote", out)

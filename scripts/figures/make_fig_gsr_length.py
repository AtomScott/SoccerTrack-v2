"""fig5_gsr_length.pdf: GS-HOTA against sequence length as a line chart.

Companion figure to Table tab:gsr_length, added per Atom's red-pen pass 2
("line chart."). Data are the GS-HOTA column of that table: nested prefixes
of match 128057, first half, one configuration, length the only variable.

Style matches make_fig3_v2.py (Okabe-Ito, DejaVu Sans 7 pt, 372 pt width).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/atom/soccertrack-v2/paper/figures"

BLUE = "#0072B2"
RED = "#D55E00"
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
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "axes.labelsize": 7,
    "axes.labelcolor": INK,
    "text.color": INK,
})

# Table tab:gsr_length: length (min), GS-HOTA. The 30 s value is the raw
# scorer output 37.1549 (score_30s_128057_1st.json) rounded once, 37.15, so
# that the figure agrees with tab:gsr_results and tab:gsr_length; the
# length-sweep CSV stores 37.155, which re-rounds to 37.16.
lengths = [0.5, 1, 2, 5, 10, 45]
gshota = [37.15, 47.59, 39.88, 30.97, 25.60, 18.09]
tick_labels = ["30 s", "1 min", "2 min", "5 min", "10 min", "45 min\n(full half)"]

fig, ax = plt.subplots(figsize=(5.15, 2.1))

ax.axvline(0.5, color=RED, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
ax.text(0.53, 20.5, "existing GSR\nbenchmark clips (30 s)", color=RED,
        fontsize=6.2, ha="left", va="bottom")

ax.plot(lengths, gshota, color=BLUE, linewidth=1.2, marker="o",
        markersize=3.2, zorder=3, clip_on=False)

for x, y in zip(lengths, gshota):
    if x == 1:
        ax.annotate(f"{y:.2f} (peak)", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color=INK,
                    fontweight="bold")
    elif x == 45:
        ax.annotate(f"{y:.2f}\n(38% of peak)", (x, y),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=6.2, color=INK)
    else:
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color=MUTED)

ax.set_xscale("log")
ax.set_xticks(lengths)
ax.set_xticklabels(tick_labels)
ax.minorticks_off()
ax.set_xlim(0.42, 54)
ax.set_ylim(0, 55)
ax.set_xlabel("sequence length (nested prefixes, log scale)")
ax.set_ylabel("GS-HOTA")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.3)
out = os.path.join(OUT_DIR, "fig5_gsr_length.pdf")
fig.savefig(out)
print("wrote", out)

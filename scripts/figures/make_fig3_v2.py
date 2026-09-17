"""fig3_stats.pdf v2: dataset statistics as a 2x2 grid, per Atom 2026-08-29.

Top row, ball action spotting:
  (a) class distribution (log-scale horizontal bars, benchmark counts)
  (b) events per match (vertical bars, asterisks on ball-clamped matches)
Bottom row, game state reconstruction:
  (c) positional density of all annotated player positions (metric pitch)
  (d) identity persistence: annotated duration per ground-truth track,
      against the 30 s clip length of existing GSR benchmarks

Designed at the sn-jnl text width (372 pt = 5.15 in) so it renders at true
size: the previous 12 in three-in-a-row design scaled to 43% and was cramped.
The FOOTPASS broadcast-visibility panel was dropped (over-indexed on FOOTPASS;
that comparison lives in the positioning subsection).

GSR inputs come from gsr_stats/ (extract_gsr_stats.py, cross-checked by
validate_extract.py). BAS counts are the benchmark numbers already in the
paper (Table tab:bas_dataset and tab:bas_per_class).
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Arc, Circle, Rectangle

SP = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(SP, "gsr_stats")
OUT_DIR = "/home/atom/soccertrack-v2/paper/figures"

from extract_gsr_stats import BINS, XR, YR  # noqa: E402  lattice-aligned bins

# Okabe-Ito (colour-blind safe)
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

# ------------------------------------------------------------- BAS panel data
BAS_COUNTS = [
    ("Pass", 9319), ("Drive", 8255), ("High Pass", 1157), ("Out", 771),
    ("Cross", 394), ("Throw In", 385), ("Ball Player Block", 353),
    ("Player Successful Tackle", 307), ("Shot", 266), ("Free Kick", 150),
    ("Goal", 44), ("Header", 31),
]
EVENTS_PER_MATCH = {
    "117092": 2086, "117093": 2252, "118575": 2117, "118576": 2131,
    "118577": 1946, "118578": 2106, "128057": 1932, "128058": 2144,
    "132831": 2440, "132877": 2278,
}
CLAMPED = {"132831", "132877"}

# ------------------------------------------------------------- GSR panel data
hist_files = sorted(glob.glob(os.path.join(STATS, "*_hist.npy")))
track_files = sorted(glob.glob(os.path.join(STATS, "*_tracks.json")))
assert len(hist_files) == 20 and len(track_files) == 20, \
    f"expected 20 halves, got {len(hist_files)} hists / {len(track_files)} tracks"

hist = np.zeros(BINS, dtype=np.int64)
for f in hist_files:
    hist += np.load(f)

durations_min = []
roles = []
n_records = 0
n_null_pitch = 0
n_out_of_range = 0
for f in track_files:
    with open(f) as fh:
        s = json.load(fh)
    n_records += s["n_object_annotations"]
    n_null_pitch += s["n_null_pitch"]
    n_out_of_range += s["n_out_of_hist_range"]
    for t in s["tracks"].values():
        durations_min.append(t["n"] / 25.0 / 60.0)
        roles.append(t["role"])
durations_min = np.asarray(durations_min)
n_tracks = durations_min.size
med = np.median(durations_min)
mean = durations_min.mean()

print(f"GSR records: {n_records:,}  positions binned: {hist.sum():,}  "
      f"null pitch: {n_null_pitch:,}  outside hist range: {n_out_of_range:,}")
print(f"tracks: {n_tracks}  duration min/median/mean/max = "
      f"{durations_min.min():.2f}/{med:.1f}/{mean:.1f}/{durations_min.max():.1f} min")
print(f"tracks >= 40 min: {(durations_min >= 40).sum()} "
      f"({(durations_min >= 40).mean()*100:.1f}%)  < 5 min: {(durations_min < 5).sum()}")
print("roles:", {r: roles.count(r) for r in set(roles)})

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(5.15, 4.55))
gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                      left=0.20, right=0.965, top=0.945, bottom=0.075,
                      hspace=0.5, wspace=0.46)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ---- (a) BAS class distribution --------------------------------------------
labels = [c[0] for c in BAS_COUNTS][::-1]  # largest at top
counts = np.array([c[1] for c in BAS_COUNTS][::-1])
y = np.arange(len(labels))
ax_a.barh(y, counts, height=0.62, color=BLUE, zorder=3)
ax_a.set_xscale("log")
ax_a.set_yticks(y, labels, fontsize=5.6)
ax_a.set_xlim(10, 60000)
ax_a.set_xlabel("events (log scale)", labelpad=1.5)
for yi, c in zip(y, counts):
    ax_a.text(c * 1.25, yi, f"{c:,}", va="center", ha="left",
              fontsize=5.6, color=INK, zorder=4)
ax_a.xaxis.grid(True, which="major", color="0.88", lw=0.5, zorder=0)
ax_a.tick_params(axis="y", length=0)
ax_a.spines[["top", "right"]].set_visible(False)

# ---- (b) BAS events per match -----------------------------------------------
matches = list(EVENTS_PER_MATCH)
# Paper labels M1 to M10: ascending provider-id order, as in Table 1.
assert matches == sorted(matches), "EVENTS_PER_MATCH must be in ascending id order"
labels = [f"M{i + 1}" for i in range(len(matches))]
ev = np.array([EVENTS_PER_MATCH[m] for m in matches])
x = np.arange(len(matches))
ax_b.bar(x, ev, width=0.62, color=BLUE, zorder=3)
ax_b.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor",
                fontsize=5.6)
for xi, m in zip(x, matches):
    if m in CLAMPED:
        ax_b.text(xi, EVENTS_PER_MATCH[m] + 40, "*", ha="center", va="bottom",
                  fontsize=9, color=RED, zorder=4)
ax_b.set_ylim(0, 2750)
ax_b.set_ylabel("annotated events", labelpad=2)
ax_b.set_xlabel("match", labelpad=1.5)
ax_b.yaxis.grid(True, color="0.88", lw=0.5, zorder=0)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.tick_params(axis="x", length=0)

# ---- (c) GSR positional density ---------------------------------------------
img = ax_c.imshow(hist.T + 1e-9, origin="lower", cmap="viridis",
                  norm=LogNorm(vmin=1, vmax=hist.max()),
                  extent=[*XR, *YR], aspect="equal", interpolation="nearest",
                  zorder=2)


def pitch_lines(ax, lw=0.5, color="white", alpha=0.85):
    z = 3
    kw = dict(lw=lw, edgecolor=color, facecolor="none", alpha=alpha, zorder=z)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, **kw))
    ax.plot([0, 0], [-34, 34], lw=lw, color=color, alpha=alpha, zorder=z)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for sgn in (-1, 1):
        # penalty area 16.5 m deep x 40.32 m wide; goal area 5.5 x 18.32
        ax.add_patch(Rectangle((sgn * 52.5, -20.16), -sgn * 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((sgn * 52.5, -9.16), -sgn * 5.5, 18.32, **kw))
        ax.add_patch(Arc((sgn * (52.5 - 11), 0), 2 * 9.15, 2 * 9.15,
                         theta1=(128.8 if sgn > 0 else -51.2),
                         theta2=(231.2 if sgn > 0 else 51.2),
                         lw=lw, color=color, alpha=alpha, zorder=z))


pitch_lines(ax_c)
ax_c.set_xlim(*XR)
ax_c.set_ylim(*YR)
ax_c.set_xticks([-52.5, 0, 52.5])
ax_c.set_yticks([-34, 0, 34])
ax_c.set_xlabel("x (m)", labelpad=1.5)
ax_c.set_ylabel("y (m)", labelpad=-1)
cb = fig.colorbar(img, ax=ax_c, fraction=0.038, pad=0.03)
cb.ax.tick_params(labelsize=5.4, length=2)
cb.outline.set_linewidth(0.4)
cb.set_label("positions per grid cell", size=5.4, labelpad=1.5)

# ---- (d) GSR identity persistence -------------------------------------------
bins = np.arange(0, 51, 2.5)
ax_d.hist(durations_min, bins=bins, color=BLUE, zorder=3)
ax_d.set_yscale("log")
ax_d.set_ylim(0.7, 700)
ax_d.set_xlim(0, 50)
ax_d.set_xlabel("annotated duration per track (min)", labelpad=1.5)
ax_d.set_ylabel("tracks (log scale)", labelpad=2)
ax_d.axvline(0.5, color=RED, ls="--", lw=0.9, zorder=4)
ax_d.text(1.6, 250, "existing GSR\nbenchmark clips\n(30 s)", fontsize=5.8,
          color=RED, va="center", ha="left")
ax_d.text(med - 1.5, 250, f"median\n{med:.0f} min", fontsize=5.8, color=INK,
          va="center", ha="right")
ax_d.yaxis.grid(True, which="major", color="0.88", lw=0.5, zorder=0)
ax_d.spines[["top", "right"]].set_visible(False)

# ---- panel letters -----------------------------------------------------------
fig.canvas.draw()
inv = fig.transFigure.inverted()
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    x0, _ = inv.transform((bb.x0, 0))
    fig.text(max(x0, 0.005), ax.get_position().y1 + 0.008, f"({letter})",
             fontsize=9, fontweight="bold", ha="left", va="bottom")

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "fig3_stats.pdf"))
fig.savefig(os.path.join(SP, "fig3_stats_v2_preview.png"), dpi=220)
print("written", os.path.join(OUT_DIR, "fig3_stats.pdf"))

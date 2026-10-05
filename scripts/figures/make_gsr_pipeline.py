"""fig6_baselines.pdf: schematics of the two baselines, one figure.

(a) The improved game state reconstruction baseline (formerly the whole of
    fig4_gsr_pipeline.pdf). White boxes: published components (the SoccerNet
    baseline structure, Somers et al. 2024). Orange-accented elements: our
    contributions (calibration fitted from the hand-annotated keypoints,
    relaxed jersey-region gate, per-detection team assignment, match-length
    fixes).
(b) The trajectory-based ball action spotting baseline (Methods, "Trajectory-
    based ball action spotting"), every stage of which is ours, so nothing in
    (b) carries the orange accent.

2026-09-24: jersey recogniser relabelled ViTSTR (was PARSeq); the
calibration box carries the orange accent the caption already claimed for
it; modification tags moved inside their boxes (the Pose and Re-ID arrows
ran through them); boxes widened where text overflowed; the hard-coded
section number dropped from the match-length strip (the caption cites it);
panel (b) added. The old fig4_gsr_pipeline.pdf is no longer written.

Designed at the sn-jnl text width (372 pt = 5.15 in) so it renders at true
size under includegraphics[width=\\linewidth]. Running the script also checks
that every label sits inside its box and fails loudly otherwise.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = "/home/atom/soccertrack-v2/paper/figures"
PREVIEW = ("/tmp/claude-1001/-home-atom-SoccerTrack-v2/"
           "5605c156-ef7c-4cfb-ac07-c28a159bbc2d/scratchpad/fix_scratch/figs")

ORANGE = "#E69F00"
INK = "#1a1a1a"
MUTED = "#666666"
EDGE = "#888888"
ARROW = "#555555"
FILL = "white"
INFILL = "#eeeeee"
OURS_FILL = "#FDF3E0"
GROUP_FILL = "#f4f4f4"

plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "text.color": INK})

# One drawing unit is the same physical length in both panels and both axes.
W = 5.15                       # figure width, in
L, R = 0.005, 0.995            # axes span, fraction of width
UNIT = W * (R - L) / 100.0     # in per unit (x runs 0..100)
H_A_UNITS, H_B_UNITS = 49.0, 45.0
LABEL = 0.20                   # in, panel label band above each panel
GAP = 0.06                     # in, between panel (a) and (b)'s label band
FIG_H = LABEL + H_A_UNITS * UNIT + GAP + LABEL + H_B_UNITS * UNIT + 0.02

fig = plt.figure(figsize=(W, FIG_H), dpi=300)


def panel_axes(y_top_in, h_units):
    h = h_units * UNIT
    ax = fig.add_axes([L, 1 - (y_top_in + h) / FIG_H, R - L, h / FIG_H])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, h_units)
    ax.axis("off")
    return ax


ax_a = panel_axes(LABEL, H_A_UNITS)
ax_b = panel_axes(LABEL + H_A_UNITS * UNIT + GAP + LABEL, H_B_UNITS)

CHECK = []   # (text artist, box patch) pairs verified after drawing


def box(ax, x0, y0, w, h, title, sub=None, tag=None, modified=False,
        fill=FILL, fs=6.3, dashed=False, edge=None):
    """Rounded box with a bold title, optional muted sub-text and an optional
    orange tag line (our modification), stacked and centred vertically."""
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.4,rounding_size=1.0",
                       facecolor=fill,
                       edgecolor=edge or (ORANGE if modified else EDGE),
                       linewidth=1.4 if modified else 0.7,
                       linestyle=(0, (3, 1.6)) if dashed else "-", zorder=3)
    ax.add_patch(p)
    # line heights in drawing units (font size in pt -> in -> units)
    pt = 1 / 72 / UNIT
    items = [(title, fs, "bold", INK)]
    if sub:
        items.append((sub, 5.4, "normal", MUTED))
    if tag:
        items.append((tag, 5.3, "bold", ORANGE))
    heights = [(s.count("\n") + 1) * f * 1.18 * pt for s, f, _, _ in items]
    gap = 0.55
    total = sum(heights) + gap * (len(items) - 1)
    y = y0 + h / 2 + total / 2
    for (s, f, wgt, col), hh in zip(items, heights):
        t = ax.text(x0 + w / 2, y - hh / 2, s, ha="center", va="center",
                    fontsize=f, fontweight=wgt, color=col, zorder=4,
                    linespacing=1.18)
        CHECK.append((t, p))
        y -= hh + gap
    return (x0, y0, w, h)


def arrow(ax, p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 color=ARROW, linewidth=0.8, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def cxy(b, side, f=0.5):
    x0, y0, w, h = b
    return {"r": (x0 + w, y0 + h * f), "l": (x0, y0 + h * f),
            "b": (x0 + w * f, y0), "t": (x0 + w * f, y0 + h)}[side]


def panel_label(ax, letter, title, note=None):
    x0 = ax.get_position().x0
    y1 = ax.get_position().y1
    t = fig.text(x0 + 0.002, y1 + 0.05 / FIG_H, f"({letter})", fontsize=9,
                 fontweight="bold", ha="left", va="bottom")
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    t2 = fig.text(bb.x1 + 0.012, y1 + 0.05 / FIG_H, title, fontsize=7,
                  ha="left", va="bottom", color=INK)
    if note:
        fig.canvas.draw()
        bb2 = t2.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb2.x1 + 0.010, y1 + 0.05 / FIG_H, note, fontsize=6.2,
                 ha="left", va="bottom", color=MUTED)


# =============================== (a) game state reconstruction =============
ax = ax_a
Y1, H = 38, 8
vid = box(ax, 1, Y1, 12, H, "Panoramic\nvideo", fill=INFILL)
det = box(ax, 17, Y1, 13, H, "Detection", "RF-DETR")
pos = box(ax, 34, Y1, 13, H, "Pose", "ViTPose")
rid = box(ax, 51, Y1, 13.5, H, "Re-ID + role", "PRTReID")
trk = box(ax, 68.5, Y1, 12.5, H, "Tracking", "BoT-SORT")
gta = box(ax, 85, Y1, 14, H, "Tracklet\nassociation", "GTA")
for a, b in ((vid, det), (det, pos), (pos, rid), (rid, trk), (trk, gta)):
    arrow(ax, cxy(a, "r"), cxy(b, "l"))

Y2, H2 = 19.5, 10
cal = box(ax, 1, Y2, 18.5, H2, "Pitch calibration",
          "annotated keypoints,\nTPS + homography", modified=True)
jer = box(ax, 23.5, Y2, 18, H2, "Jersey number", "torso crop + ViTSTR",
          tag="gate 500 → 100 px²", modified=True)
tea = box(ax, 45, Y2, 20.5, H2, "Team side", "2-means clustering",
          tag="+ per-detection variant", modified=True)
vot = box(ax, 69, Y2, 15.5, H2, "Attribute\nvoting", "per tracklet")
gst = box(ax, 88, Y2 - 2.5, 11, H2 + 5, "Game\nstate", "positions +\nidentities",
          fill=INFILL)

arrow(ax, cxy(vid, "b"), cxy(cal, "t", 0.35))
arrow(ax, cxy(pos, "b"), cxy(jer, "t"), rad=0.12)
arrow(ax, cxy(rid, "b"), cxy(tea, "t"), rad=0.12)
arrow(ax, cxy(jer, "r"), cxy(tea, "l"))
arrow(ax, cxy(tea, "r"), cxy(vot, "l"))
arrow(ax, cxy(gta, "b"), cxy(vot, "t", 0.6), rad=-0.12)
arrow(ax, cxy(vot, "r"), cxy(gst, "l"))
YL = 13.2
# calibration output joins the game state: one elbow line, one arrowhead
xc = cxy(cal, "b")[0]
ax.plot([xc, xc, 86.3], [Y2 - 0.4, YL, YL], color=ARROW, lw=0.8, zorder=2,
        solid_joinstyle="miter")
arrow(ax, (86.3, YL), (91.0, Y2 - 2.5), rad=-0.25)

# ---- bottom strip: match-length fixes
ax.add_patch(FancyBboxPatch((1, 2), 81, 7.5,
                            boxstyle="round,pad=0.4,rounding_size=1.0",
                            facecolor=OURS_FILL, edgecolor=ORANGE,
                            linewidth=1.4, zorder=3))
ax.text(41.5, 7.4, "Match-length fixes", fontsize=6.0,
        fontweight="bold", ha="center", va="center", color=INK, zorder=4)
ax.text(41.5, 4.2,
        "batch-local result merging  ·  grouped consistency check  ·  "
        "separable association similarity  ·  malloc_trim",
        fontsize=5.4, ha="center", va="center", color=MUTED, zorder=4)
ax.text(84.5, 5.7, "37 h per half,\none 16 GB GPU", fontsize=5.4, color=MUTED,
        ha="left", va="center", zorder=4)

# =============================== (b) ball action spotting ===================
ax = ax_b
# ---- inputs
trj = box(ax, 1, 29, 13.5, 13, "Player\ntrajectories",
          "ground truth,\n22 players,\npitch coordinates", fill=INFILL)
bal = box(ax, 1, 20.5, 13.5, 6, "Ball track", "optional", fill=INFILL,
          dashed=True)

# ---- feature stage: one frame, four player-feature groups + ball features
FX0, FX1, FY0, FY1 = 18.5, 99, 19.5, 43.5
ax.add_patch(FancyBboxPatch((FX0, FY0), FX1 - FX0, FY1 - FY0,
                            boxstyle="round,pad=0.4,rounding_size=1.0",
                            facecolor=GROUP_FILL, edgecolor=EDGE,
                            linewidth=0.7, zorder=1))
ax.text(FX0 + 1.0, FY1 - 1.9,
        "Per-frame features, sampled at 5 Hz", fontsize=6.3, fontweight="bold",
        ha="left", va="center", zorder=4)
ax.text(FX1 - 1.0, FY1 - 1.9, "82 from player positions",
        fontsize=5.8, color=MUTED, ha="right", va="center", zorder=4)
GY, GH = 28.5, 11.2
groups = [
    (18.0, "All players", "centroid,\ndispersion,\nconvex hull"),
    (18.0, "Ball proxy", "soft minimum\nover opposing\nplayer pairs"),
    (20.5, "Per team", "centroid, dispersion,\nhull, speed, defensive\nand attacking lines"),
    (18.0, "Density grid", "6 × 3 bilinear\nplayer density,\nper team"),
]
gap_g = (FX1 - FX0 - 2.0 - sum(g[0] for g in groups)) / (len(groups) - 1)
gx = FX0 + 1.0
gboxes = []
for w, t, s in groups:
    gboxes.append(box(ax, gx, GY, w, GH, t, s, fs=6.0))
    gx += w + gap_g
bf = box(ax, FX0 + 1.0, 21.0, FX1 - FX0 - 2.0, 4.6,
         "+ 19 ball-track features (optional): position, velocity, speed, "
         "acceleration,\nnearest player of each team, players within 3, 5, "
         "10 m, distances to goals and boundaries", fs=5.4, dashed=True)
# the long ball-feature line is one bold title; make it regular weight
CHECK[-1][0].set_fontweight("normal")
CHECK[-1][0].set_color(MUTED)

# into the feature stage as a whole (all groups use the player positions)
arrow(ax, cxy(trj, "r", 0.55), (FX0 - 0.4, cxy(trj, "r", 0.55)[1]))
arrow(ax, cxy(bal, "r"), (FX0 + 0.6, 21.0 + 2.3))

# ---- temporal convolutional network, sigmoids, decoding, output
BY, BH = 1.5, 11.0
ax.add_patch(FancyBboxPatch((0.6, 0.6), 54.4, 15.0,
                            boxstyle="round,pad=0.4,rounding_size=1.0",
                            facecolor=GROUP_FILL, edgecolor=EDGE,
                            linewidth=0.7, zorder=1))
ax.text(54.6, 14.25, "Temporal convolutional network", fontsize=6.0,
        fontweight="bold", ha="right", va="center", color=INK, zorder=4)
inp = box(ax, 1.6, BY, 13.4, BH, "Input\nprojection", "pointwise,\n64 channels")
res = box(ax, 18.5, BY, 20, BH, "Residual blocks × 6",
          "dilations 1, 2, 4, 8, 16, 32\nnon-causal, receptive\nfield ±25 s")
out = box(ax, 42, BY, 12.4, BH, "Output\nprojection", "pointwise,\n12 logits\nper frame")
sig = box(ax, 58.5, BY, 11, BH, "Sigmoids", "independent,\none per class")
pk = box(ax, 72.5, BY, 15.5, BH, "Peak picking",
         "per-class maxima\nabove a floor,\nwithin-class NMS")
spt = box(ax, 91, BY, 8, BH, "Ranked\nspots", "per\nclass", fill=INFILL)

arrow(ax, (FX0 + 3.0, FY0 - 0.4), cxy(inp, "t", 0.62), rad=0.25)
for a, b in ((inp, res), (res, out), (out, sig), (sig, pk), (pk, spt)):
    arrow(ax, cxy(a, "r"), cxy(b, "l"))

# ---- panel labels
panel_label(ax_a, "a", "Game state reconstruction baseline")
panel_label(ax_b, "b", "Ball action spotting baseline", "every stage ours")

# ---- overflow check: each label must sit well inside its box
fig.canvas.draw()
r = fig.canvas.get_renderer()
bad = []
for t, p in CHECK:
    tb, pb = t.get_window_extent(r), p.get_window_extent(r)
    m = 1.0 * fig.dpi / 72          # 1 pt in display px
    # >= 2 pt clear of the (rounded) border sideways, >= 1 pt vertically
    worst = min((tb.x0 - pb.x0) / 2, (pb.x1 - tb.x1) / 2,
                tb.y0 - pb.y0, pb.y1 - tb.y1) / m
    if worst < 1.0:
        bad.append((t.get_text().replace("\n", " | "), round(worst, 2)))
if bad:
    for s, w in bad:
        print(f"OVERFLOW ({w} pt margin): {s}")
    raise SystemExit("text overflows its box; fix the layout")
print(f"overflow check: {len(CHECK)} labels inside their boxes")

fig.savefig(f"{OUT_DIR}/fig6_baselines.pdf")
if os.path.isdir(PREVIEW):
    fig.savefig(f"{PREVIEW}/fig6_baselines_preview.png", dpi=220)
print("saved", f"{OUT_DIR}/fig6_baselines.pdf")

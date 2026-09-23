"""fig6_calibration.pdf: released pixels -> undistorted view -> metric pitch.

Methods figure added per Atom's red-pen pass 2 ("Deserves an image: before
calib -> after calib; coordinate proj -> on-field etc.", Methods 4.2/4.3).

Frame: match 117093, first half, CLPD-117093-1st clip frame 2600, whose
Labels-GameState.json annotations are aligned with the jpgs by construction.
The homography is refit from the 65 hand-annotated calibrated keypoints
(cv2.findHomography, median residual 2.9 px); GT pitch coords map to the
image as corner = (x + 52.5, y + 34), verified against bbox_image with
median error 13.8 px over the 22 players (the other three sign conventions
give 227 px and worse).

NOTE data finding, do not silently "fix": empirically +y points to the NEAR
touchline (camera side) in 117093, while Methods 4.2 says y increases
towards the FAR touchline. Flagged to Atom; panel (c) draws the near side
at the bottom, matching the camera view.

Style matches make_fig3_v2.py (Okabe-Ito, DejaVu Sans 7 pt, 372 pt width).
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, Circle, ConnectionPatch, Rectangle

RAW_DIR = "/data/share/SoccerTrack-v2/data/raw/117093"
CLIP = "/data/share/SoccerTrack-v2/SoccerNetGS/valid/CLPD-117093-1st"
FRAME = 2600
OUT = "/home/atom/soccertrack-v2/paper/figures/fig6_calibration.pdf"

BLUE = "#0072B2"
ORANGE = "#E69F00"
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

# ------------------------------------------------------------------- data
raw = cv2.cvtColor(cv2.imread(f"{CLIP}/img1/{FRAME:06d}.jpg"), cv2.COLOR_BGR2RGB)
mapx = np.load(f"{RAW_DIR}/117093_mapx.npy")
mapy = np.load(f"{RAW_DIR}/117093_mapy.npy")

ck = json.load(open(f"{RAW_DIR}/117093_calibrated_keypoints.json"))
P = np.array([eval(k) for k in ck], float)
I = np.array(list(ck.values()), float)
H, _ = cv2.findHomography(P, I, cv2.RANSAC, 5.0)

labels = json.load(open(f"{CLIP}/Labels-GameState.json"))
iid = f"2903{FRAME:06d}"
players = [a for a in labels["annotations"]
           if a.get("supercategory") == "object" and a["image_id"] == iid
           and a.get("bbox_pitch")]

def proj(pts):
    """corner-origin pitch coords (n,2) -> calibrated image px (n,2)"""
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    q = (H @ ph.T).T
    return q[:, :2] / q[:, 2:3]

# pitch template polylines in corner-origin coords (105 x 68)
def template_lines():
    L = []
    L.append([(0, 0), (105, 0), (105, 68), (0, 68), (0, 0)])        # outer
    L.append([(52.5, 0), (52.5, 68)])                                # halfway
    for x0, w in [(0, 16.5), (105 - 16.5, 16.5)]:                    # penalty
        xa, xb = (0, 16.5) if x0 == 0 else (105 - 16.5, 105)
        L.append([(xa, 13.84), (xb, 13.84), (xb, 54.16), (xa, 54.16)]
                 if x0 == 0 else
                 [(xb, 13.84), (xa, 13.84), (xa, 54.16), (xb, 54.16)])
    for xa, xb in [(0, 5.5), (105 - 5.5, 105)]:                      # goal area
        L.append([(xa, 24.84), (xb, 24.84), (xb, 43.16), (xa, 43.16)]
                 if xa == 0 else
                 [(xb, 24.84), (xa, 24.84), (xa, 43.16), (xb, 43.16)])
    th = np.linspace(0, 2 * np.pi, 80)                               # circle
    L.append(list(zip(52.5 + 9.15 * np.cos(th), 34 + 9.15 * np.sin(th))))
    return L

# --------------------------------------------------------------- figure
# Full-width layout per Atom 2026-08-31 ("fix fig 6 to be full width, at
# least for a and b"): panels (a) and (b) span the exact canvas width, with
# panel heights derived from the image aspect ratios; (c) stays centred.
# Axes are placed manually and the canvas is saved without bbox trimming, so
# includegraphics[width=\linewidth] renders the images at full text width.
#
# 2026-09-24 legibility pass: (b) was a 0.54 in strip of the whole corrected
# frame, in which the pitch is ~190 px tall and 3,000 px wide, so it printed
# as an illegible sliver. (b) is now the pitch region only (a strip showing
# the whole template), plus an enlarged detail of the boxed region holding
# every player. Both are resampled from the released frame through the
# released undistortion maps at 2x / 3x the maps' own output density: at the
# centre of the pitch the corrected frame is ~3x coarser than the raw frame,
# so the detail recovers real resolution rather than upscaling. Template and
# positions are drawn as vector overlays, and rasters are written at 330 ppi
# at the printed width (>= 300 ppi required).
W = 5.15
DPI = 330
# calibrated-image pixel windows (x0, x1, y0, y1); pitch corners project to
# x 498..3453, y 404..592, and the 22 players to x 1531..2262, y 415..568
STRIP = (460, 3490, 380, 615)
DETAIL = (1450, 2350, 385, 602)
H_A = W * raw.shape[0] / raw.shape[1]                          # (a) full frame
H_S = W * (STRIP[3] - STRIP[2]) / (STRIP[1] - STRIP[0])        # (b) pitch strip
H_D = W * (DETAIL[3] - DETAIL[2]) / (DETAIL[1] - DETAIL[0])    # (b) detail
ZOOM_GAP = 0.13
H_C = 1.85                                          # (c) axes box height
W_C = H_C * 112.0 / 77.0                            # from (c)'s data limits
TITLE, GAP, TOP, BOT = 0.17, 0.10, 0.04, 0.30
FIG_H = (TOP + TITLE + H_A + GAP + TITLE + H_S + ZOOM_GAP + H_D + GAP + TITLE
         + H_C + BOT)
fig = plt.figure(figsize=(W, FIG_H))


def add_full_ax(y_top, h):
    return fig.add_axes([0.0, 1 - (y_top + h) / FIG_H, 1.0, h / FIG_H])


y = TOP + TITLE

# (a) released frame
ax_a = add_full_ax(y, H_A)
y += H_A + GAP + TITLE
ax_a.imshow(raw)
ax_a.set_axis_off()
ax_a.set_title("(a)  released panoramic frame (distorted)",
               fontsize=7, loc="left", pad=3)

# (b) distortion-corrected frame, pitch template and annotated positions
fmapx, fmapy = cv2.convertMaps(mapx, mapy, cv2.CV_32FC1)


def undistort(win, scale):
    """Corrected frame over calibrated-pixel window win, sampled `scale` times
    denser than the maps' own grid: the (smooth) maps are interpolated at the
    sub-pixel positions, then the raw frame is sampled there."""
    x0, x1, y0, y1 = win
    gu, gv = np.meshgrid(x0 + np.arange((x1 - x0) * scale) / scale,
                         y0 + np.arange((y1 - y0) * scale) / scale)
    gu, gv = gu.astype(np.float32), gv.astype(np.float32)
    sx = cv2.remap(fmapx, gu, gv, cv2.INTER_LINEAR)
    sy = cv2.remap(fmapy, gu, gv, cv2.INTER_LINEAR)
    img = cv2.remap(raw, sx, sy, cv2.INTER_CUBIC)
    half = 0.5 / scale  # extent in calibrated px, pixel centres on the grid
    ext = (x0 - half, x0 + gu.shape[1] / scale - half,
           y0 + gu.shape[0] / scale - half, y0 - half)
    return img, ext


dense_lines = []
for line in template_lines():
    pts = np.array(line, float)
    dense = [p0 + t * (p1 - p0) for p0, p1 in zip(pts[:-1], pts[1:])
             for t in np.linspace(0, 1, 60)]
    dense_lines.append(proj(np.array(dense)))
pos = proj(np.array([[a["bbox_pitch"]["x_bottom_middle"] + 52.5,
                      a["bbox_pitch"]["y_bottom_middle"] + 34] for a in players]))
cols = [BLUE if a["attributes"]["team"] == "left" else ORANGE for a in players]


def draw_b(ax, win, scale, lw, ms, mew):
    img, ext = undistort(win, scale)
    ax.imshow(img, extent=ext, interpolation="antialiased")
    for q in dense_lines:
        ax.plot(q[:, 0], q[:, 1], color=RED, lw=lw, solid_capstyle="round",
                zorder=3)
    for (u, v), col in zip(pos, cols):
        ax.plot(u, v, "o", ms=ms, mec="white", mew=mew, color=col, zorder=4)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_axis_off()


ax_s = add_full_ax(y, H_S)
y += H_S + ZOOM_GAP
draw_b(ax_s, STRIP, 2, lw=0.55, ms=2.3, mew=0.35)
ax_s.set_title("(b)  after distortion correction: pitch template and annotated"
               " positions projected into the image",
               fontsize=7, loc="left", pad=3)
ax_d = add_full_ax(y, H_D)
y += H_D + GAP + TITLE
draw_b(ax_d, DETAIL, 3, lw=0.8, ms=3.6, mew=0.5)
mag = (STRIP[1] - STRIP[0]) / (DETAIL[1] - DETAIL[0])
ax_d.text(0.006, 0.955, f"boxed region, enlarged {mag:.1f}×",
          transform=ax_d.transAxes, fontsize=5.8, color="white", ha="left",
          va="top", zorder=6,
          bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))
# box on the strip and connectors to the detail
ax_s.add_patch(Rectangle((DETAIL[0], DETAIL[2]), DETAIL[1] - DETAIL[0],
                         DETAIL[3] - DETAIL[2], fill=False, ec="white",
                         lw=0.7, zorder=5))
for xs, xd in ((DETAIL[0], 0.0), (DETAIL[1], 1.0)):
    fig.add_artist(ConnectionPatch(
        xyA=(xs, DETAIL[3]), coordsA=ax_s.transData,
        xyB=(xd, 1.0), coordsB=ax_d.transAxes,
        color=MUTED, lw=0.5, zorder=1))

# (c) metric pitch, centred
ax_c = fig.add_axes([(1 - W_C / W) / 2, 1 - (y + H_C) / FIG_H, W_C / W, H_C / FIG_H])
for line in template_lines():
    pts = np.array(line, float)
    ax_c.plot(pts[:, 0] - 52.5, pts[:, 1] - 34, color=MUTED, lw=0.6)
ax_c.plot(0, 0, ".", color=MUTED, ms=2)
ax_c.plot([-41.5], [0], ".", color=MUTED, ms=2)
ax_c.plot([41.5], [0], ".", color=MUTED, ms=2)
for a in players:
    bp = a["bbox_pitch"]
    col = BLUE if a["attributes"]["team"] == "left" else ORANGE
    ax_c.plot(bp["x_bottom_middle"], bp["y_bottom_middle"], "o", ms=3.4,
              mec="white", mew=0.4, color=col)
ax_c.set_xlim(-56, 56)
ax_c.set_ylim(38.5, -38.5)  # near touchline (camera side, +y) at the bottom
ax_c.set_aspect("equal")
ax_c.set_xticks([-52.5, 0, 52.5])
ax_c.set_yticks([-34, 0, 34])
ax_c.set_xlabel("x (m)", labelpad=1)
ax_c.set_ylabel("y (m)", labelpad=-4)
ax_c.set_title("(c)  the same instant in metric pitch coordinates",
               fontsize=7, loc="left", pad=3)
for s in ("top", "right"):
    ax_c.spines[s].set_visible(False)

fig.savefig(OUT, dpi=DPI)
PREVIEW = ("/tmp/claude-1001/-home-atom-SoccerTrack-v2/"
           "5605c156-ef7c-4cfb-ac07-c28a159bbc2d/scratchpad/fix_scratch/figs")
if os.path.isdir(PREVIEW):
    fig.savefig(os.path.join(PREVIEW, "fig6_calibration_preview.png"), dpi=250)
print("wrote", OUT)

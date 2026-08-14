"""Per-frame trajectory features for ball action spotting, from GSR player positions.

Experiment B of docs/experiment-design-bas.md: predict ball events from the game state
alone -- 22 player positions on the pitch -- with no pixels and no ball. The GSR ground
truth contains no ball at all (20 outfield players + 2 goalkeepers), so every feature here
is derived from player configuration and motion.

WHAT THE COORDINATE QUANTISATION FORCES
    Positions are quantised to 1.05 m in x and 0.68 m in y (they derive from a normalised
    pitch position stored to two decimals). A one-frame finite difference therefore carries
    1.05/0.04 = 26 m/s of pure quantisation noise. Measured on 128057's first half, the
    median apparent player speed is:

        window   0.08 s   0.24 s   0.40 s   0.96 s   1.60 s
        median   0.00     2.83     1.70     1.42     1.43   m/s
        quant.   13.12    4.38     2.62     1.09     0.66   m/s

    Below ~1 s the "speed" is the quantiser, not the player: at 0.08 s the median is exactly
    zero (the position has not changed) while the 95th percentile is 13.13 m/s (it has moved
    exactly one step). Velocity is therefore computed at TWO scales -- a short one that
    still localises an event in time and a long one that is physically meaningful -- and
    both are given to the model rather than one being chosen for it.

FEATURE GROUPS (see FEATURE_NAMES for the exact layout)
    global      centroid, dispersion and convex hull of all 22 players
    contest     a soft-minimum over opposing pairs, which is the best available proxy for
                where the ball is when there is no ball: its location, its motion, how many
                players are converging on it, and how fast they are moving
    per team    centroid, dispersion, hull, speed, and the defensive/attacking line
    occupancy   a coarse 6x3 soft-binned player-density grid per team, which encodes the
                configuration without depending on any player ordering

    Nothing here uses bbox_image. Those boxes are auto-generated and are not ground truth.
    Nothing here uses the ball, the event annotations, or anything not present in the
    released GSR files for every match.
"""
from __future__ import annotations

import numpy as np

# Pitch, from <pitch width="105" height="68"/> in every match's metadata.
PITCH_L, PITCH_W = 105.0, 68.0
FPS = 25

# Velocity half-windows in frames. 6 -> 0.48 s (localises), 19 -> 1.52 s (physical).
VEL_SHORT, VEL_LONG = 6, 19

# Occupancy grid, per team.
GRID_X, GRID_Y = 6, 3

# Softness of the contest point, in metres. Opposing pairs are weighted exp(-d/tau), so a
# pair 2 m further apart than the closest contributes 1/e as much. Small enough that the
# contest point still tracks the tightest duel, large enough that it never hinges on which
# of two equidistant pairs an argmin happened to pick.
CONTEST_TAU = 2.0


def _hull_area(pts: np.ndarray) -> float:
    """Convex hull area of a small 2-D point set, by Andrew's monotone chain + shoelace."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(p) < 3:
        return 0.0

    def half(points):
        out: list = []
        for q in points:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    hull = half(p) + half(p[::-1])
    if len(hull) < 3:
        return 0.0
    h = np.array(hull)
    return float(abs(np.dot(h[:, 0], np.roll(h[:, 1], -1)) -
                     np.dot(h[:, 1], np.roll(h[:, 0], -1))) / 2.0)


def _soft_grid(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Player density on a GRID_X x GRID_Y grid, bilinearly spread so it varies smoothly.

    A hard histogram jumps by a whole count when a player crosses a cell edge, which puts a
    step change into the input for no physical reason. Bilinear spreading removes that.
    """
    g = np.zeros((GRID_X, GRID_Y), np.float32)
    if xs.size == 0:
        return g.ravel()
    u = np.clip((xs / PITCH_L + 0.5) * (GRID_X - 1), 0, GRID_X - 1)
    v = np.clip((ys / PITCH_W + 0.5) * (GRID_Y - 1), 0, GRID_Y - 1)
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    i1, j1 = np.minimum(i0 + 1, GRID_X - 1), np.minimum(j0 + 1, GRID_Y - 1)
    fu, fv = u - i0, v - j0
    for i, j, w in ((i0, j0, (1 - fu) * (1 - fv)), (i1, j0, fu * (1 - fv)),
                    (i0, j1, (1 - fu) * fv), (i1, j1, fu * fv)):
        np.add.at(g, (i, j), w)
    return g.ravel()


def _dense_grids(track_npz: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(X, Y, team) arrays of shape (n_frames+1, n_entities), NaN where unobserved."""
    frame = track_npz["frame"]
    pid = track_npz["player_id"]
    n_f = int(frame.max())
    uniq = np.unique(pid)
    col = {int(p): i for i, p in enumerate(uniq.tolist())}
    idx = np.array([col[int(p)] for p in pid.tolist()], np.int32)
    X = np.full((n_f + 1, uniq.size), np.nan, np.float32)
    Y = np.full_like(X, np.nan)
    team = np.full(uniq.size, -1, np.int8)
    X[frame - 1, idx] = track_npz["x"]
    Y[frame - 1, idx] = track_npz["y"]
    team[idx] = track_npz["team"]
    return X, Y, team, n_f


def _velocity(A: np.ndarray, k: int) -> np.ndarray:
    """Central finite difference over +/-k frames, in metres per second."""
    out = np.full_like(A, np.nan)
    dt = 2 * k / FPS
    out[k:-k] = (A[2 * k:] - A[:-2 * k]) / dt
    return out


def _names() -> list[str]:
    n = ["cen_x", "cen_y", "cen_vx", "cen_vy", "std_x", "std_y", "hull", "span_x", "span_y"]
    n += ["ct_x", "ct_y", "ct_vx", "ct_vy", "ct_mindist",
          "ct_pairs_2m", "ct_pairs_4m", "ct_pairs_8m",
          "ct_near_5m", "ct_near_10m", "ct_localspeed"]
    for side in ("L", "R"):
        n += [f"{side}_cen_x", f"{side}_cen_y", f"{side}_cen_vx", f"{side}_cen_vy",
              f"{side}_std_x", f"{side}_std_y", f"{side}_hull",
              f"{side}_meanspeed", f"{side}_maxspeed",
              f"{side}_min_x", f"{side}_max_x", f"{side}_ct_dist", f"{side}_n"]
    for side in ("L", "R"):
        n += [f"{side}_grid{i}" for i in range(GRID_X * GRID_Y)]
    return n


FEATURE_NAMES = _names()
N_FEATURES = len(FEATURE_NAMES)


def build_features(track_npz, stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, frames).

    features : (T, N_FEATURES) float32, one row per sampled frame
    frames   : (T,) int32, the 1-based GSR frame index of each row

    ``stride`` subsamples in time. The default 5 gives 5 Hz (200 ms), comfortably finer
    than the 1 s evaluation tolerance while cutting the sequence length fivefold.
    """
    X, Y, team, n_f = _dense_grids(track_npz)
    VXs, VYs = _velocity(X, VEL_SHORT), _velocity(Y, VEL_SHORT)
    VXl, VYl = _velocity(X, VEL_LONG), _velocity(Y, VEL_LONG)

    left = np.where(team == 0)[0]
    right = np.where(team == 1)[0]
    rows = np.arange(0, n_f, stride)
    out = np.zeros((rows.size, N_FEATURES), np.float32)
    # Frames where a team is briefly unobserved give empty nanmean/nanmax slices; those
    # rows are zero-filled at the end, so the warnings carry no information.
    np.seterr(invalid="ignore")

    # The contest point is tracked across sampled rows so its velocity can be differenced.
    ct_hist = np.full((rows.size, 2), np.nan, np.float32)

    for t, f in enumerate(rows):
        xs, ys = X[f], Y[f]
        ok = ~np.isnan(xs)
        c = 0

        # ---- global -------------------------------------------------------
        gx, gy = xs[ok], ys[ok]
        if gx.size == 0:
            out[t] = np.nan
            continue
        out[t, c:c + 2] = (gx.mean(), gy.mean()); c += 2
        vx, vy = VXl[f][ok], VYl[f][ok]
        out[t, c:c + 2] = (np.nanmean(vx) if vx.size else 0.0,
                           np.nanmean(vy) if vy.size else 0.0); c += 2
        out[t, c:c + 2] = (gx.std(), gy.std()); c += 2
        out[t, c] = _hull_area(np.stack([gx, gy], 1)); c += 1
        out[t, c:c + 2] = (gx.ptp(), gy.ptp()); c += 2

        # ---- contest: the closest opposing pair ---------------------------
        li = left[~np.isnan(xs[left])]
        ri = right[~np.isnan(xs[right])]
        if li.size and ri.size:
            d = np.hypot(xs[li][:, None] - xs[ri][None, :],
                         ys[li][:, None] - ys[ri][None, :])
            # SOFT-minimum midpoint, not the argmin pair. Positions are quantised to 1.05 m,
            # so the closest opposing pair is exactly TIED on 6.7% of frames; an argmin then
            # picks between equally valid pairs on a tie-break, which puts arbitrary jitter
            # into the ball proxy and made the feature fail its own reflection test. The
            # softmin is tie-free, varies smoothly as players move, and is exactly
            # equivariant under reflection.
            w = np.exp(-d / CONTEST_TAU)
            wsum = w.sum()
            mx = (xs[li][:, None] + xs[ri][None, :]) / 2.0
            my = (ys[li][:, None] + ys[ri][None, :]) / 2.0
            ctx = float((w * mx).sum() / wsum)
            cty = float((w * my).sum() / wsum)
            mind = float(d.min())
            p2, p4, p8 = (d < 2).sum(), (d < 4).sum(), (d < 8).sum()
        else:
            ctx, cty, mind, p2, p4, p8 = gx.mean(), gy.mean(), np.nan, 0, 0, 0
        ct_hist[t] = (ctx, cty)
        out[t, c:c + 2] = (ctx, cty); c += 2
        c += 2  # ct_vx, ct_vy are filled after the loop, by differencing ct_hist
        out[t, c] = mind; c += 1
        out[t, c:c + 3] = (p2, p4, p8); c += 3
        dct = np.hypot(gx - ctx, gy - cty)
        near5, near10 = dct < 5, dct < 10
        out[t, c:c + 2] = (near5.sum(), near10.sum()); c += 2
        sp = np.hypot(VXl[f][ok], VYl[f][ok])
        out[t, c] = float(np.nanmean(sp[near10])) if near10.any() else 0.0; c += 1

        # ---- per team -----------------------------------------------------
        for side in (li, ri):
            if side.size == 0:
                c += 13
                continue
            sx, sy = xs[side], ys[side]
            out[t, c:c + 2] = (sx.mean(), sy.mean()); c += 2
            out[t, c:c + 2] = (np.nanmean(VXl[f][side]), np.nanmean(VYl[f][side])); c += 2
            out[t, c:c + 2] = (sx.std(), sy.std()); c += 2
            out[t, c] = _hull_area(np.stack([sx, sy], 1)); c += 1
            ssp = np.hypot(VXs[f][side], VYs[f][side])
            out[t, c:c + 2] = (np.nanmean(ssp), np.nanmax(ssp) if ssp.size else 0.0); c += 2
            out[t, c:c + 2] = (sx.min(), sx.max()); c += 2
            out[t, c] = float(np.hypot(sx.mean() - ctx, sy.mean() - cty)); c += 1
            out[t, c] = side.size; c += 1

        # ---- occupancy ----------------------------------------------------
        for side in (li, ri):
            out[t, c:c + GRID_X * GRID_Y] = _soft_grid(xs[side], ys[side])
            c += GRID_X * GRID_Y

    # Contest-point velocity, differenced over the sampled series (+/-1 row). This is NOT
    # a player velocity: the contest point teleports whenever a different opposing pair
    # becomes the closest one, which reaches 147 m/s unclipped. The jump is informative --
    # it is what a long ball looks like without a ball -- but left unbounded it would
    # dominate feature normalisation, so it is clipped to a generous 30 m/s.
    iv = FEATURE_NAMES.index("ct_vx")
    dt = 2 * stride / FPS
    out[1:-1, iv] = np.clip((ct_hist[2:, 0] - ct_hist[:-2, 0]) / dt, -30, 30)
    out[1:-1, iv + 1] = np.clip((ct_hist[2:, 1] - ct_hist[:-2, 1]) / dt, -30, 30)

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out, (rows + 1).astype(np.int32)

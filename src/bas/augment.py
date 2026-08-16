"""Reflection augmentation for trajectory features.

WHY
    There are only twelve training halves. The first attempt overfitted hard -- training
    loss halved while validation loss doubled by epoch 6 -- because 170,000 correlated rows
    from six matches is not much data for a model with a 25-second receptive field.

    Football is symmetric under reflection: mirroring the pitch end-to-end, or top-to-
    bottom, maps a valid game state to another valid game state. Neither reflection changes
    any of the twelve event labels, because the label set is team-agnostic (there is no
    "pass to the left"), so both are pure input augmentations with the targets untouched.
    Together they give a 4x enlargement for free.

WHY IT IS DONE IN FEATURE SPACE
    Re-deriving 82 features from mirrored raw tracks would cost the same 20 s per half as
    the original build, on every epoch. Applying a permutation and a sign vector to the
    finished feature matrix is a single fused operation.

    That is only safe if the permutation is right, and an index error here would silently
    feed the model corrupted inputs -- the sort of defect that shows up as "the model just
    does not work". tests/test_bas_augment.py therefore mirrors the RAW TRACKS, rebuilds
    the features through the normal path, and requires the result to equal the cheap
    feature-space transform. If any index is wrong, it fails.
"""
from __future__ import annotations

import numpy as np

from src.bas.features import (BALL_NAMES, FEATURE_NAMES, FEATURE_NAMES_BALL,
                              GRID_X, GRID_Y, N_FEATURES, N_FEATURES_BALL)

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# Per-team scalar features, in the order they appear in FEATURE_NAMES.
_TEAM_KEYS = ("cen_x", "cen_y", "cen_vx", "cen_vy", "std_x", "std_y", "hull",
              "meanspeed", "maxspeed", "min_x", "max_x", "ct_dist", "n")


def _grid_perm(flip_x: bool, flip_y: bool) -> np.ndarray:
    """Permutation of a raveled GRID_X x GRID_Y occupancy grid under reflection."""
    out = np.empty(GRID_X * GRID_Y, np.int64)
    for i in range(GRID_X):
        for j in range(GRID_Y):
            si = (GRID_X - 1 - i) if flip_x else i
            sj = (GRID_Y - 1 - j) if flip_y else j
            out[i * GRID_Y + j] = si * GRID_Y + sj
    return out


def _build(flip_x: bool, flip_y: bool, width: int = N_FEATURES):
    """(source, sign, offset) with mirrored[:,k] = sign[k]*original[:,source[k]] + offset[k].

    The offset exists for one column only: `b_near_is_right` is an indicator of which team
    the ball's nearest player belongs to, and mirroring end-to-end exchanges the teams, so
    it maps to 1 - v rather than +/-v.
    """
    src = np.arange(width, dtype=np.int64)
    sgn = np.ones(width, np.float32)
    off = np.zeros(width, np.float32)

    def negate(name):
        sgn[_IDX[name]] = -1.0

    # ---- global and contest: pure sign flips on the mirrored axis ----------
    if flip_x:
        for n in ("cen_x", "cen_vx", "ct_x", "ct_vx"):
            negate(n)
    if flip_y:
        for n in ("cen_y", "cen_vy", "ct_y", "ct_vy"):
            negate(n)
    # std, hull, spans, distances, counts and speeds are reflection-invariant.

    # ---- per-team blocks --------------------------------------------------
    # Mirroring x exchanges which team occupies the left, so the two blocks swap. The block
    # label then still means "the team on the left", which is what the model was trained on.
    for a, b in (("L", "R"), ("R", "L")):
        other = b if flip_x else a
        for key in _TEAM_KEYS:
            dst = _IDX[f"{a}_{key}"]
            if flip_x and key == "min_x":
                # after x -> -x the minimum becomes the negated maximum
                src[dst] = _IDX[f"{other}_max_x"]
                sgn[dst] = -1.0
            elif flip_x and key == "max_x":
                src[dst] = _IDX[f"{other}_min_x"]
                sgn[dst] = -1.0
            else:
                src[dst] = _IDX[f"{other}_{key}"]
                if key in ("cen_x", "cen_vx") and flip_x:
                    sgn[dst] = -1.0
                elif key in ("cen_y", "cen_vy") and flip_y:
                    sgn[dst] = -1.0

    # ---- occupancy grids --------------------------------------------------
    gp = _grid_perm(flip_x, flip_y)
    for a, b in (("L", "R"), ("R", "L")):
        other = b if flip_x else a
        base_dst = _IDX[f"{a}_grid0"]
        base_src = _IDX[f"{other}_grid0"]
        for k in range(GRID_X * GRID_Y):
            src[base_dst + k] = base_src + int(gp[k])
            sgn[base_dst + k] = 1.0

    # ---- ball block, when present ----------------------------------------
    if width > N_FEATURES:
        bi = {n: N_FEATURES + i for i, n in enumerate(BALL_NAMES)}
        if flip_x:
            for n in ("b_x", "b_vx"):
                sgn[bi[n]] = -1.0
            # end-to-end reflection exchanges the teams
            src[bi["b_near_is_right"]] = bi["b_near_is_right"]
            sgn[bi["b_near_is_right"]] = -1.0
            off[bi["b_near_is_right"]] = 1.0
            for a, b in (("b_nearL_dist", "b_nearR_dist"), ("b_nearR_dist", "b_nearL_dist")):
                src[bi[a]] = bi[b]
            for a, b in (("b_dist_goal_left", "b_dist_goal_right"),
                         ("b_dist_goal_right", "b_dist_goal_left")):
                src[bi[a]] = bi[b]
        if flip_y:
            for n in ("b_y", "b_vy"):
                sgn[bi[n]] = -1.0
        # speed, acceleration, counts, and every distance are reflection-invariant
    return src, sgn, off


_TRANSFORMS = {(fx, fy, w): _build(fx, fy, w)
               for fx in (False, True) for fy in (False, True)
               for w in (N_FEATURES, N_FEATURES_BALL)}


def mirror(feat: np.ndarray, flip_x: bool = False, flip_y: bool = False) -> np.ndarray:
    """Reflect a (T, N_FEATURES) or (T, N_FEATURES_BALL) feature matrix."""
    if not (flip_x or flip_y):
        return feat
    w = feat.shape[1]
    if (flip_x, flip_y, w) not in _TRANSFORMS:
        raise ValueError(f"no mirror transform for width {w}; expected "
                         f"{N_FEATURES} or {N_FEATURES_BALL}")
    src, sgn, off = _TRANSFORMS[(flip_x, flip_y, w)]
    return feat[:, src] * sgn + off


def random_mirror(feat: np.ndarray, rng) -> np.ndarray:
    return mirror(feat, bool(rng.integers(2)), bool(rng.integers(2)))

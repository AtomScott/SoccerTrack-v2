"""Reflection augmentation must equal rebuilding the features from mirrored tracks.

WHY THIS EXISTS
    src/bas/augment.py mirrors a finished 82-column feature matrix with a permutation and a
    sign vector, because re-deriving features from mirrored raw tracks would cost 20 s per
    half on every epoch. That shortcut is only valid if every index is right, and a wrong
    index would not raise -- it would quietly feed the model corrupted inputs and show up
    much later as "the trajectory model does not work very well".

    So the test does the expensive thing once: it mirrors the RAW TRACKS, pushes them
    through the ordinary feature builder, and requires the result to match the cheap
    transform. Any error in the grid permutation, the min_x/max_x exchange, the left/right
    team swap or a sign will fail it.

Run:
    .venv/bin/python -m pytest tests/test_bas_augment.py -q -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRACKS = Path("/data/share/SoccerTrack-v2/data/derived/bas/tracks")
SAMPLE = TRACKS / "128057_1st_tracks.npz"

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(),
    reason=f"{SAMPLE} not built; run scripts/bas/extract_tracks.py")


def _load_window(n_frames: int = 6000) -> dict:
    """A slice of one half, small enough to rebuild features from quickly."""
    d = np.load(SAMPLE)
    m = d["frame"] <= n_frames
    return {k: d[k][m] for k in ("frame", "track_id", "player_id", "team", "role", "x", "y")}


def _mirror_tracks(t: dict, flip_x: bool, flip_y: bool) -> dict:
    out = {k: v.copy() for k, v in t.items()}
    if flip_x:
        out["x"] = -out["x"]
        # Mirroring end-to-end exchanges which team occupies the left half, so the team
        # labels must be exchanged for the state to remain self-consistent.
        swapped = out["team"].copy()
        swapped[out["team"] == 0] = 1
        swapped[out["team"] == 1] = 0
        out["team"] = swapped
    if flip_y:
        out["y"] = -out["y"]
    return out


@pytest.mark.parametrize("flip_x,flip_y", [(True, False), (False, True), (True, True)])
def test_feature_space_mirror_matches_rebuild(flip_x, flip_y):
    from src.bas.augment import mirror
    from src.bas.features import FEATURE_NAMES, build_features

    tracks = _load_window()
    base, _ = build_features(tracks, stride=25)
    rebuilt, _ = build_features(_mirror_tracks(tracks, flip_x, flip_y), stride=25)
    cheap = mirror(base, flip_x=flip_x, flip_y=flip_y)

    assert cheap.shape == rebuilt.shape
    # Hull area and the soft grid involve floating-point geometry, so compare with a
    # tolerance scaled to each column rather than exactly.
    scale = np.maximum(np.abs(rebuilt).max(axis=0), 1.0)
    err = np.abs(cheap - rebuilt).max(axis=0) / scale
    worst = int(np.argmax(err))
    print(f"flip_x={flip_x} flip_y={flip_y}: worst column {FEATURE_NAMES[worst]} "
          f"rel err {err[worst]:.2e}")
    bad = [(FEATURE_NAMES[i], float(err[i])) for i in np.where(err > 1e-4)[0]]
    assert not bad, f"feature-space mirror disagrees with a rebuild on: {bad}"


def test_mirror_is_an_involution():
    from src.bas.augment import mirror
    from src.bas.features import build_features

    base, _ = build_features(_load_window(3000), stride=25)
    for fx, fy in ((True, False), (False, True), (True, True)):
        back = mirror(mirror(base, fx, fy), fx, fy)
        assert np.allclose(back, base, atol=1e-5), f"mirror({fx},{fy}) is not its own inverse"


def test_identity_is_a_no_op():
    from src.bas.augment import mirror
    from src.bas.features import build_features

    base, _ = build_features(_load_window(3000), stride=25)
    assert mirror(base, False, False) is base


if __name__ == "__main__":
    for fx, fy in ((True, False), (False, True), (True, True)):
        test_feature_space_mirror_matches_rebuild(fx, fy)
    test_mirror_is_an_involution()
    test_identity_is_a_no_op()
    print("PASS")

"""GSR loader test: the released files must actually parse.

WHY THIS EXISTS
    `_parse_gsr` originally did `records = json.loads(...)` and then iterated the result as a
    list. Every released GSR file is a JSON OBJECT (SoccerNet GameState), so that iteration
    yielded the top-level KEY STRINGS and `r["image_id"]` raised

        TypeError: string indices must be integers

    on all 20 files. The loader could not read a single one, and nothing caught it because no
    test ever pointed it at real data. This test does.

    It also pins the two facts that make the released layout surprising: `image_id` is a string
    whose numeric suffix is 1-based, and the pitch position lives in `bbox_pitch`'s
    bottom-middle keys rather than in `x`/`y` fields.

Run:
    .venv/bin/python -m pytest tests/test_gsr_loader.py -q -s
    .venv/bin/python tests/test_gsr_loader.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Resolve THIS checkout's src/, not whatever the venv's editable install points at.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("DATA", "/data/share/SoccerTrack-v2/data"))
GSR = DATA / "production" / "gsr"
MATCH = os.environ.get("GSR_TEST_MATCH", "128057")
# The released halves are ~2.7 GB. Slicing a small window keeps this test fast while still
# exercising the real on-disk structure rather than a hand-made fixture.
N_FRAMES = int(os.environ.get("GSR_TEST_FRAMES", "40"))


def _slice_to_tmp() -> Path | None:
    """Write a small but genuine GameState file: real structure, fewer frames."""
    src = GSR / MATCH / f"{MATCH}_1st.json"
    if not src.exists():
        return None
    d = json.loads(src.read_text())
    images = sorted(d["images"], key=lambda im: int(str(im["image_id"]).split("_")[-1]))[:N_FRAMES]
    keep = {im["image_id"] for im in images}
    out = {
        "info": dict(d["info"]),
        "images": images,
        "annotations": [a for a in d["annotations"] if a.get("image_id") in keep],
        "categories": d["categories"],
    }
    out["info"]["seq_length"] = len(images)
    tmp = Path(tempfile.mkdtemp(prefix="gsr-loader-")) / f"{MATCH}_1st.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out))
    return tmp


def test_released_gamestate_parses():
    """The headline regression: this raised TypeError on every real file before the fix."""
    from src.data_utils.soccertrack_v2 import _parse_gsr

    p = _slice_to_tmp()
    if p is None:
        print(f"SKIP: {GSR} not present")
        return
    frames = list(_parse_gsr(p, half=1))
    assert frames, "parsed zero frames from a real GameState file"
    assert len(frames) == N_FRAMES, f"expected {N_FRAMES} frames, got {len(frames)}"

    # The released GSR ground truth holds exactly 22 entities in every frame: 20 outfield
    # players plus 2 goalkeepers, with no ball and no referees.
    counts = {len(f.entities) for f in frames}
    assert counts == {22}, f"expected 22 entities in every frame, saw {sorted(counts)}"

    roles = {p_.role for f in frames for p_ in f.entities}
    assert roles <= {"player", "goalkeeper"}, f"unexpected roles: {roles}"

    # image_id is 1-based in the released layout, so the first frame is 1 rather than 0.
    assert frames[0].image_id == 1, f"expected a 1-based first frame, got {frames[0].image_id}"
    print(f"parsed {len(frames)} frames, {len(frames[0].entities)} entities each, "
          f"roles={sorted(roles)}, first image_id={frames[0].image_id}")


def test_pitch_positions_are_sane():
    """x/y must come from bbox_pitch's bottom-middle keys, in centre-origin metres."""
    from src.data_utils.soccertrack_v2 import _parse_gsr

    p = _slice_to_tmp()
    if p is None:
        print(f"SKIP: {GSR} not present")
        return
    frames = list(_parse_gsr(p, half=1))
    xs = [e.x for f in frames for e in f.entities]
    ys = [e.y for f in frames for e in f.entities]
    assert xs and ys
    # centre-origin on a 105x68 pitch, with a little slack for players over the line
    assert max(abs(v) for v in xs) <= 60, f"x out of range: {min(xs)}..{max(xs)}"
    assert max(abs(v) for v in ys) <= 40, f"y out of range: {min(ys)}..{max(ys)}"
    # a real frame has players spread out, not stacked on one point
    assert max(xs) - min(xs) > 5, "players are implausibly co-located in x"
    print(f"x in [{min(xs):.1f}, {max(xs):.1f}]  y in [{min(ys):.1f}, {max(ys):.1f}]")


def test_flat_record_layout_still_works():
    """Predictions use the flat layout; dispatch must handle both without a filename hint."""
    from src.data_utils.soccertrack_v2 import _parse_gsr

    recs = [
        {"image_id": 0, "track_id": 1, "role": "player", "jersey_number": 9,
         "team_side": "left", "x": 1.5, "y": -2.0},
        {"image_id": 0, "track_id": 2, "role": "goalkeeper", "jersey_number": None,
         "team_side": "right", "x": -40.0, "y": 0.5},
        {"image_id": 1, "track_id": 1, "role": "player", "jersey_number": 9,
         "team_side": "left", "x": 1.6, "y": -2.1},
    ]
    tmp = Path(tempfile.mkdtemp(prefix="gsr-flat-")) / "x_1st.json"
    tmp.write_text(json.dumps(recs))
    frames = list(_parse_gsr(tmp, half=1))
    assert len(frames) == 2, f"expected 2 frames, got {len(frames)}"
    assert frames[0].image_id == 0, "flat records are 0-based and must stay that way"
    assert len(frames[0].entities) == 2
    assert frames[0].entities[0].jersey_number == 9
    print(f"flat layout: {len(frames)} frames, first image_id={frames[0].image_id}")


if __name__ == "__main__":
    test_released_gamestate_parses()
    test_pitch_positions_are_sane()
    test_flat_record_layout_still_works()
    print("PASS")

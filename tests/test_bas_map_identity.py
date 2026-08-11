"""BAS plumbing test: scoring ground truth against itself must give mAP == 1.0.

Same principle as tests/test_gs_hota_identity.py. If the parser or the AP computation is
wrong, a perfect "prediction" will not score 1.0.

This test would have caught all four defects fixed alongside it, each of which made BAS
evaluation impossible rather than merely inaccurate:

  1. the event array is keyed "actions", not "annotations"  -> KeyError on every file
  2. labels are UPPER CASE, not Title Case                  -> ValueError on every event
  3. three matches carry a third period whose gameTime omits the half prefix ("90:01")
                                                            -> ValueError on unpack
  4. `position` is absolute from match start, not per-half, so Event.image_id put every
     half-2 event 67,500 frames late

The BAS files are small (a few hundred KB), so unlike the GSR test this runs over all ten
matches directly with no slicing.

Run:
    .venv/bin/python -m pytest tests/test_bas_map_identity.py -q -s
    .venv/bin/python tests/test_bas_map_identity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve THIS checkout's src/, not whatever the venv's editable install points at. The
# project is installed in editable mode against a different working tree, so without this a
# test run here silently exercises that other tree's code.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("DATA", "/data/share/SoccerTrack-v2/data"))
BAS = DATA / "production" / "bas"
MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]


def _available() -> list[str]:
    return [m for m in MATCHES if (BAS / m / f"{m}_12_class_events.json").exists()]


def test_every_match_parses():
    """All ten must parse. As shipped, not one of them could be read."""
    from src.data_utils.soccertrack_v2 import BAS_LABELS, _parse_bas

    matches = _available()
    assert matches, f"no BAS files under {BAS}"
    total = 0
    for m in matches:
        events = list(_parse_bas(BAS / m / f"{m}_12_class_events.json"))
        assert events, f"{m}: parsed zero events"
        total += len(events)
        for e in events:
            assert e.label in BAS_LABELS, f"{m}: uncanonicalised label {e.label!r}"
            assert e.half >= 1
    print(f"parsed {total} events across {len(matches)} matches")
    assert total > 20000, f"expected >20k events across the dataset, got {total}"


def test_gt_against_itself_scores_one():
    from src.evaluation.bas_map import score_many

    matches = _available()
    res = score_many(BAS, BAS, matches)
    for m in matches:
        for tol in (1, 5):
            got = res[m][f"mAP@{tol}s"]
            assert got > 0.999, f"{m} mAP@{tol}s = {got}, expected ~1.0 scoring GT against itself"
    print(f"mAP@1s = {res['overall']['mAP@1s']:.4f}   "
          f"mAP@5s = {res['overall']['mAP@5s']:.4f}   over {len(matches)} matches")


def test_position_is_absolute_not_per_half():
    """Guard the finding that `position` is absolute from match start.

    docs/format-bas.md and the paper both say it is relative to the half's kickoff. It is
    not: half-2 events do not restart near zero. If a future data release changes this, the
    per-half offset in Event.t_ms_in_half becomes wrong and this test should fail loudly.
    """
    from src.data_utils.soccertrack_v2 import HALF_MS, _parse_bas

    events = list(_parse_bas(BAS / "117093" / "117093_12_class_events.json"))
    h2 = [e for e in events if e.half == 2]
    assert h2, "no half-2 events"
    assert min(e.t_ms for e in h2) > HALF_MS * 0.9, (
        "half-2 `position` values appear to restart near zero, i.e. they are per-half after "
        "all. Event.t_ms_in_half and Event.image_id must be revisited."
    )
    # ...and the derived per-half offset must land near the start of the half.
    assert min(e.t_ms_in_half for e in h2) < 60_000


if __name__ == "__main__":
    test_every_match_parses()
    test_gt_against_itself_scores_one()
    test_position_is_absolute_not_per_half()
    print("PASS")

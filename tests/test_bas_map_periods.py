"""The evaluator must exclude events that have no imagery and no tracks.

WHY THIS EXISTS
    Three of the ten matches (117092, 132831, 132877) were played as THREE 45-minute
    periods. 2,232 events -- 9.4% of all annotations, and 22.8% of test match 132831 -- sit
    in a third period for which no video and no GSR file was ever produced. They are ground
    truth that nothing can match, so they depress recall by an amount that differs per
    match: a model scored on 128057 and 132831 together is penalised on one of the two for
    a reason that has nothing to do with the model.

    Neither existing BAS test can catch this. test_bas_map_identity.py scores ground truth
    against itself, so the third-period events match themselves and AP stays 1.0.
    test_bas_map_ranking.py builds synthetic events that have no period structure at all.
    This is the same class of blind spot that hid three defects in the GS-HOTA prediction
    path: a test whose construction cannot express the failure will never see it.

THE DISCRIMINATING CONSTRUCTION
    Score a REALISTIC partial prediction -- ground truth restricted to periods 1 and 2, the
    most a trajectory or video model could possibly emit -- against the full released file.
    With the filter on, that prediction is perfect and scores 1.0. With the filter off, the
    third-period events are unmatchable and the score must fall. If the two are equal, the
    filter is not doing anything.

Run:
    .venv/bin/python -m pytest tests/test_bas_map_periods.py -q -s
    .venv/bin/python tests/test_bas_map_periods.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

DATA = Path("/data/share/SoccerTrack-v2/data")
GT_ROOT = DATA / "production" / "bas"
# 132831 has a third period; 128057 does not. Together they are the test split, so this is
# exactly the pair the paper's numbers are computed on.
MATCHES = ["128057", "132831"]

pytestmark = pytest.mark.skipif(
    not GT_ROOT.exists(), reason=f"dataset not mounted at {GT_ROOT}")


def _write_period_1_2_prediction(dst: Path) -> dict[str, int]:
    """A perfect prediction that only covers the periods a model can actually see."""
    from src.data_utils.bas_periods import is_evaluable, load_table

    table = load_table()
    counts = {}
    for mid in MATCHES:
        src = GT_ROOT / mid / f"{mid}_12_class_events.json"
        acts = json.loads(src.read_text())["actions"]
        mp = table[mid]["periods"]
        keep = [a for a in acts if is_evaluable(a["gameTime"], int(a["position"]), mp)]
        for i, a in enumerate(keep):
            a["score"] = 1.0 - i * 1e-9  # any strictly decreasing ranking
        out = dst / mid
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{mid}_12_class_events.json").write_text(
            json.dumps({"match_id": mid, "fps": 25.0, "actions": keep}))
        counts[mid] = len(acts) - len(keep)
    return counts


def test_third_period_events_are_excluded(tmp_path):
    from src.evaluation.bas_map import score_many

    dropped = _write_period_1_2_prediction(tmp_path)
    print(f"third-period events per match: {dropped}")
    assert dropped["132831"] > 0, (
        "132831 is supposed to have third-period events; if this fails the period table is "
        "stale -- regenerate it with scripts/bas/audit_annotations.py --write-periods")
    assert dropped["128057"] == 0, "128057 has only two periods"

    on = score_many(tmp_path, GT_ROOT, MATCHES, (1,), filter_periods=True)
    off = score_many(tmp_path, GT_ROOT, MATCHES, (1,), filter_periods=False)

    a = on["overall"]["mAP@1s"]
    b = off["overall"]["mAP@1s"]
    print(f"mAP@1s with the period filter    : {a:.4f}  (n_gt {on['overall']['n_gt']})")
    print(f"mAP@1s without the period filter : {b:.4f}  (n_gt {off['overall']['n_gt']})")

    assert a > 0.999, (
        f"a prediction covering every evaluable event scored {a}, expected ~1.0 -- the "
        "filter is dropping events it should keep")
    assert b < a - 1e-6, (
        f"the filter changes nothing: {a} with, {b} without. Third-period events are still "
        "being scored, and every model will be penalised for annotations it cannot see.")


def test_the_penalty_lands_on_the_right_match(tmp_path):
    """Without the filter the damage is per-match, which is why it must not be averaged away."""
    from src.evaluation.bas_map import score_many

    _write_period_1_2_prediction(tmp_path)
    off = score_many(tmp_path, GT_ROOT, MATCHES, (1,), filter_periods=False)
    clean, dirty = off["128057"]["mAP@1s"], off["132831"]["mAP@1s"]
    print(f"unfiltered mAP@1s -- 128057 (two periods) {clean:.4f}, "
          f"132831 (three periods) {dirty:.4f}")
    assert clean > 0.999, "128057 has no third period, so it should be unaffected"
    assert dirty < clean - 1e-6, (
        "132831 should be penalised and 128057 should not; if they are equal the filter's "
        "absence is not being detected per match")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_third_period_events_are_excluded(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_the_penalty_lands_on_the_right_match(Path(d))
    print("PASS")

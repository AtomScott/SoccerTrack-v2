"""BAS evaluator must rank predictions by confidence, not by time.

WHY THIS EXISTS
    `_map_per_class` used to sort predictions by `(half, t_ms)`, discarding any confidence
    ordering, in contradiction of the contract stated inside `_ap_tolerant` ("callers: sort by
    confidence desc"). That is not average precision -- it is a time-ordered precision-recall
    traversal, in which a detector gains nothing from ranking its output well and is not
    penalised for emitting a flood of low-confidence spots.

    tests/test_bas_map_identity.py cannot catch this. Scoring ground truth against itself makes
    every prediction a true positive, so AP is 1.0 in whatever order they arrive.

    The discriminating construction below is: take the ground truth, then add false positives.
    Rank them BELOW the true events and AP should stay high. Rank them ABOVE and AP must drop.
    If those two scores come out equal, ranking is being ignored.

Run:
    .venv/bin/python -m pytest tests/test_bas_map_ranking.py -q -s
    .venv/bin/python tests/test_bas_map_ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABEL = "Pass"


def _events(n_true: int = 20, n_false: int = 20, fp_score: float = 0.9,
            tp_score: float = 0.5):
    """Ground truth plus false positives, with configurable relative confidence."""
    from src.data_utils.soccertrack_v2 import Event

    gt = [Event(half=1, clock="00:00", t_ms=10_000 * (i + 1), label=LABEL)
          for i in range(n_true)]
    # true positives: exactly on the GT times
    tp = [Event(half=1, clock="00:00", t_ms=e.t_ms, label=LABEL, score=tp_score) for e in gt]
    # false positives: far from any GT event (well beyond a 5 s tolerance)
    fp = [Event(half=1, clock="00:00", t_ms=10_000 * (i + 1) + 5_000, label=LABEL,
                score=fp_score) for i in range(n_false)]
    return gt, tp, fp


def _ap(pred, gt, tol_s: int = 1) -> float:
    from src.evaluation.bas_map import _map_per_class
    return _map_per_class(pred, gt, tol_ms=tol_s * 1000)[LABEL]


def test_ranking_changes_the_score():
    """The core guard: good ranking must beat bad ranking."""
    gt, tp, fp = _events()

    # false positives ranked BELOW the true positives
    good = _ap(tp + [e.__class__(**{**e.__dict__, "score": 0.1}) for e in fp], gt)
    # false positives ranked ABOVE the true positives
    bad = _ap([e.__class__(**{**e.__dict__, "score": 0.99}) for e in fp] + tp, gt)

    print(f"AP with false positives ranked last  : {good:.4f}")
    print(f"AP with false positives ranked first : {bad:.4f}")
    assert good > bad + 1e-6, (
        f"ranking is being ignored: good={good} bad={bad}. The evaluator must sort predictions "
        "by descending Event.score, not by time."
    )


def test_perfect_prediction_still_scores_one():
    """The fix must not break the case the identity test covers."""
    gt, tp, _ = _events()
    ap = _ap(tp, gt)
    print(f"AP for a perfect prediction: {ap:.6f}")
    assert ap > 0.999, f"a perfect prediction scored {ap}, expected ~1.0"


def test_order_is_deterministic_without_scores():
    """Ground truth carries no scores; the result must still be stable, not arbitrary."""
    gt, tp, _ = _events()
    unscored = [e.__class__(**{**e.__dict__, "score": None}) for e in tp]
    a = _ap(unscored, gt)
    b = _ap(list(reversed(unscored)), gt)
    print(f"AP unscored, forward {a:.6f} / reversed {b:.6f}")
    assert abs(a - b) < 1e-9, "unscored predictions must fall back to a deterministic order"


if __name__ == "__main__":
    test_ranking_changes_the_score()
    test_perfect_prediction_still_scores_one()
    test_order_is_deterministic_without_scores()
    print("PASS")

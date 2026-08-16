"""The two aggregation steps are load-bearing and were previously unverified.

WHY THIS EXISTS
    `score_many` does two things beyond per-class AP, and both change the headline number:

      * it POOLS detections across matches before computing AP, rather than averaging each
        match's mAP. For a class with ten instances split five and five, the two give
        materially different answers, and pooling is what the SoccerNet protocol does.
      * it computes a SUPPORT-WEIGHTED mean alongside the unweighted one, because the class
        imbalance is 301x and an unweighted mean over twelve classes hands Header, with 31
        instances in the whole dataset, the same vote as Pass with 9,319.

    Neither had a test. Every defect this evaluator has turned out to have lived in a code
    path that no test exercised, so "it looks right" is not the standard being applied here.

Run:
    .venv/bin/python -m pytest tests/test_bas_map_aggregation.py -q -s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FPS = 25


def _write(root: Path, match: str, events: list[tuple[int, str, float | None]]):
    """events: (t_ms, label, score). score None writes ground truth."""
    d = root / match
    d.mkdir(parents=True, exist_ok=True)
    acts = []
    for t, lab, sc in events:
        a = {"gameTime": f"1 - {t // 60000:02d}:{(t // 1000) % 60:02d}",
             "label": lab, "position": str(t), "team": None}
        if sc is not None:
            a["score"] = sc
        acts.append(a)
    (d / f"{match}_12_class_events.json").write_text(
        json.dumps({"match_id": match, "fps": 25.0, "actions": acts}))


def _periods(matches, n_frames=200_000):
    return {m: {"periods": {"1": {"t0_ms": 0.0, "n_frames": n_frames,
                                  "t0_source": "measured_vs_actor_nearest_ball"},
                            "2": {"t0_ms": 2_700_000.0, "n_frames": n_frames,
                                  "t0_source": "measured_vs_actor_nearest_ball"},
                            "3": {"t0_ms": None, "n_frames": None}}}
            for m in matches}


def test_pooling_is_not_a_mean_of_per_match_scores(tmp_path):
    """Construct a case where pooled AP and mean-of-per-match-mAP must differ.

    Match A is predicted perfectly; match B's single prediction is confident and wrong. Under
    pooling, that confident false positive outranks A's true positives and depresses the
    whole curve. Under per-match averaging, A still scores 1.0 and the damage is halved.
    """
    from src.evaluation.bas_map import score_many

    gt_root, pr_root = tmp_path / "gt", tmp_path / "pred"
    A = [(10_000 + i * 10_000, "Pass", None) for i in range(8)]
    B = [(10_000 + i * 10_000, "Pass", None) for i in range(8)]
    _write(gt_root, "A", A); _write(gt_root, "B", B)
    # A: every event hit, modest confidence. B: one very confident prediction far from any GT.
    _write(pr_root, "A", [(t, "Pass", 0.5) for t, _, _ in A])
    _write(pr_root, "B", [(5_000, "Pass", 0.99)])

    sc = score_many(pr_root, gt_root, ["A", "B"], (1,), periods=_periods(["A", "B"]))
    pooled = sc["overall"]["mAP@1s"]
    per_match_mean = np.mean([sc["A"]["mAP@1s"], sc["B"]["mAP@1s"]])
    print(f"A alone {sc['A']['mAP@1s']:.4f}, B alone {sc['B']['mAP@1s']:.4f}")
    print(f"pooled {pooled:.4f}  vs  mean of per-match {per_match_mean:.4f}")

    assert sc["A"]["mAP@1s"] > 0.99, "match A is predicted perfectly"
    assert sc["B"]["mAP@1s"] == 0.0, "match B has no true positive"
    assert abs(pooled - per_match_mean) > 1e-6, (
        "pooled AP equals the mean of per-match mAPs, which means detections are not being "
        "pooled before the AP is computed")


def test_weighted_mean_tracks_support(tmp_path):
    """The weighted mean must follow the large class, and the unweighted must not."""
    from src.evaluation.bas_map import score_many

    gt_root, pr_root = tmp_path / "gt", tmp_path / "pred"
    # 20 Pass events (predicted perfectly) and 2 Header events (missed entirely).
    gt = [(10_000 + i * 10_000, "Pass", None) for i in range(20)]
    gt += [(500_000, "Header", None), (520_000, "Header", None)]
    _write(gt_root, "A", gt)
    _write(pr_root, "A", [(t, "Pass", 0.9) for t, lab, _ in gt if lab == "Pass"])

    sc = score_many(pr_root, gt_root, ["A"], (1,), periods=_periods(["A"]))["overall"]
    macro, weighted = sc["mAP@1s"], sc["mAPw@1s"]
    print(f"support: Pass {sc['support']['Pass']}, Header {sc['support']['Header']}")
    print(f"per class: Pass {sc['perClass@1s']['Pass']:.4f}, "
          f"Header {sc['perClass@1s']['Header']:.4f}")
    print(f"macro {macro:.4f}   support-weighted {weighted:.4f}")

    assert sc["perClass@1s"]["Pass"] > 0.99 and sc["perClass@1s"]["Header"] == 0.0
    # Only two classes have support, so the macro mean is the plain average of the two.
    assert abs(macro - 0.5 * (sc["perClass@1s"]["Pass"] + 0.0)) < 1e-9
    # 20 of 22 instances are Pass, so the weighted mean must sit near Pass's AP.
    assert weighted > 0.85, (
        f"weighted mean {weighted} should be near Pass's AP given Pass is 20 of 22 "
        "instances; it is being computed unweighted or weighted by the wrong quantity")
    assert weighted > macro + 0.3, "the two means must separate when support is skewed"


def test_pooled_matching_never_crosses_matches(tmp_path):
    """A prediction in one match must not satisfy a ground-truth event in another.

    Both matches carry a Pass at the same timestamp. Predicting it only in match A must leave
    match B's event unmatched; if pooling flattened the match key, one prediction would
    satisfy both and recall would be double-counted.
    """
    from src.evaluation.bas_map import score_many

    gt_root, pr_root = tmp_path / "gt", tmp_path / "pred"
    _write(gt_root, "A", [(60_000, "Pass", None)])
    _write(gt_root, "B", [(60_000, "Pass", None)])
    _write(pr_root, "A", [(60_000, "Pass", 0.9)])
    _write(pr_root, "B", [])

    sc = score_many(pr_root, gt_root, ["A", "B"], (1,), periods=_periods(["A", "B"]))
    o = sc["overall"]
    print(f"pooled: n_gt {o['n_gt']}, n_pred {o['n_pred']}, AP {o['perClass@1s']['Pass']:.4f}")
    assert o["n_gt"] == 2 and o["n_pred"] == 1
    # One of two ground-truth events found: recall caps at 0.5, so 11-point AP is ~6/11.
    assert 0.4 < o["perClass@1s"]["Pass"] < 0.6, (
        f"pooled AP {o['perClass@1s']['Pass']} implies the single prediction matched both "
        "matches' ground truth; matching must be keyed by match")


if __name__ == "__main__":
    import tempfile

    for fn in (test_pooling_is_not_a_mean_of_per_match_scores,
               test_weighted_mean_tracks_support,
               test_pooled_matching_never_crosses_matches):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("PASS")

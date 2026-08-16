"""Our tolerant mAP must equal SoccerNet's, because the paper cites their protocol.

WHY THIS EXISTS
    The evaluator inherited an 11-point interpolated AP that looked like SoccerNet's and was
    not. Reading their source (SoccerNet/Evaluation/ActionSpotting.py) turned up three
    divergences, one of which changes the numbers substantially:

      1. THE TOLERANCE IS HALVED. Their matching condition is
             abs(pred_index - gt_index) <= delta / 2
         where `delta` is `tolerance_seconds * framerate`. So SoccerNet's "mAP@1s" accepts a
         prediction within +/- 0.5 s of the ground truth, not +/- 1 s. Ours accepted +/- 1 s
         and was therefore twice as permissive.

      2. MATCHING RUNS FROM GROUND TRUTH, NOT FROM PREDICTIONS. For each ground-truth event
         they take the HIGHEST-SCORING unmatched prediction inside the window. We took, for
         each prediction in confidence order, the NEAREST unmatched ground truth. The two
         disagree whenever a window holds several candidates -- which, at a Pass every 2.4 s,
         is common.

      3. THE PRECISION-RECALL CURVE IS SAMPLED AT 200 FIXED CONFIDENCE THRESHOLDS
         (`np.linspace(0, 1, 200)`), not at every prediction.

    This module vendors a faithful port of their algorithm and requires our evaluator to
    agree with it. A test that merely checked our own implementation against itself would
    have been satisfied by the wrong answer -- the same blind spot that let the identity test
    pass while the ranking was being discarded.

REFERENCE
    SoccerNet-v3 ActionSpotting evaluation, functions `compute_class_scores`,
    `compute_precision_recall_curve` and `compute_mAP`. Ported here at frame resolution with
    the same 200-threshold grid and the same 11-point interpolation.

Run:
    .venv/bin/python -m pytest tests/test_bas_map_soccernet_parity.py -q -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FPS = 25


# ---------------------------------------------------------------------------
# Faithful port of SoccerNet's algorithm, on dense per-frame arrays
# ---------------------------------------------------------------------------

def sn_compute_class_scores(target, detection, delta):
    """SoccerNet's per-class TP/FP assignment. `delta` is in frames; window is delta/2."""
    gt_indexes = np.where(target != 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    game_detections = np.zeros((len(pred_indexes), 2))
    game_detections[:, 0] = np.copy(pred_scores)

    remove_indexes = []
    for gt_index in gt_indexes:
        max_score = -1
        max_index = None
        game_index = 0
        selected_game_index = 0
        for pred_index, pred_score in zip(pred_indexes, pred_scores):
            if pred_index < gt_index - delta:
                game_index += 1
                continue
            if pred_index > gt_index + delta:
                break
            if (abs(pred_index - gt_index) <= delta / 2 and pred_score > max_score
                    and pred_index not in remove_indexes):
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1
        if max_index is not None:
            game_detections[selected_game_index, 1] = 1
            remove_indexes.append(max_index)
    return game_detections, len(gt_indexes)


def sn_ap(target, detection, delta):
    """SoccerNet AP for one class: 200-threshold PR curve, 11-point interpolation."""
    dets, n_gt = sn_compute_class_scores(target, detection, delta)
    if n_gt == 0:
        return float("nan")
    precision, recall = [], []
    for threshold in np.linspace(0, 1, 200):
        idx = np.where(dets[:, 0] >= threshold)[0]
        tp = float(np.sum(dets[idx, 1]))
        precision.append(np.nan_to_num(tp / len(idx)) if len(idx) else 0.0)
        recall.append(np.nan_to_num(tp / n_gt))
    precision, recall = np.array(precision), np.array(recall)
    order = np.argsort(recall)
    precision, recall = precision[order], recall[order]
    ap = 0.0
    for j in np.arange(11) / 10:
        m = recall >= j
        ap += float(np.max(precision[m])) if m.any() else 0.0
    return ap / 11


# ---------------------------------------------------------------------------
# Bridge: our Event lists -> SoccerNet's dense arrays
# ---------------------------------------------------------------------------

def _dense(gt_ms, pred_ms, pred_scores, n_frames):
    target = np.zeros(n_frames)
    for t in gt_ms:
        target[int(round(t / 1000 * FPS))] = 1
    detection = np.full(n_frames, -1.0)
    for t, s in zip(pred_ms, pred_scores):
        i = int(round(t / 1000 * FPS))
        detection[i] = max(detection[i], s)
    return target, detection


def _rand_case(rng, n_gt=40, n_pred=90, span_s=600):
    gt = np.sort(rng.choice(np.arange(50, span_s * FPS - 50), n_gt, replace=False))
    pr = np.sort(rng.choice(np.arange(50, span_s * FPS - 50), n_pred, replace=False))
    sc = rng.uniform(0.01, 0.99, n_pred)
    return gt * 1000 / FPS, pr * 1000 / FPS, sc, span_s * FPS


@pytest.mark.parametrize("tol_s", [1, 5])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_our_ap_matches_soccernet(tol_s, seed):
    """Same events, same tolerance -> same AP, to 1e-9."""
    from src.data_utils.soccertrack_v2 import Event
    from src.evaluation.bas_map import _map_per_class

    rng = np.random.default_rng(seed)
    gt_ms, pr_ms, sc, n_frames = _rand_case(rng)
    target, detection = _dense(gt_ms, pr_ms, sc, n_frames)
    expected = sn_ap(target, detection, delta=tol_s * FPS)

    gt = [Event(half=1, clock="", t_ms=int(t), label="Pass") for t in gt_ms]
    pred = [Event(half=1, clock="", t_ms=int(t), label="Pass", score=float(s))
            for t, s in zip(pr_ms, sc)]
    got = _map_per_class(pred, gt, tol_ms=tol_s * 1000)["Pass"]

    print(f"tol={tol_s}s seed={seed}: ours {got:.6f}  SoccerNet {expected:.6f}  "
          f"diff {abs(got - expected):.2e}")
    assert abs(got - expected) < 1e-9, (
        f"our AP {got} != SoccerNet's {expected} at tol={tol_s}s. The paper cites their "
        "protocol, so the two must agree exactly.")


def test_tolerance_is_half_width_not_full():
    """SoccerNet's @1s accepts +/-0.5 s. A prediction 0.75 s out must NOT be a true positive.

    This is the divergence that mattered: the original implementation accepted +/-1 s and so
    scored a strictly easier task than the protocol the paper cites.
    """
    from src.data_utils.soccertrack_v2 import Event
    from src.evaluation.bas_map import _map_per_class

    gt = [Event(half=1, clock="", t_ms=100_000, label="Pass")]
    near = [Event(half=1, clock="", t_ms=100_000 + 400, label="Pass", score=0.9)]
    far = [Event(half=1, clock="", t_ms=100_000 + 750, label="Pass", score=0.9)]

    ap_near = _map_per_class(near, gt, tol_ms=1000)["Pass"]
    ap_far = _map_per_class(far, gt, tol_ms=1000)["Pass"]
    print(f"AP with prediction 0.40 s out: {ap_near:.4f}")
    print(f"AP with prediction 0.75 s out: {ap_far:.4f}")
    assert ap_near > 0.99, "a prediction 0.40 s out is inside +/-0.5 s and must match"
    assert ap_far == 0.0, (
        "a prediction 0.75 s out is outside SoccerNet's +/-0.5 s window at tol=1s and must "
        "not match; accepting it means the reported tolerance is twice the protocol's")


if __name__ == "__main__":
    for s in range(5):
        for t in (1, 5):
            test_our_ap_matches_soccernet(t, s)
    test_tolerance_is_half_width_not_full()
    print("PASS")

"""BAS tolerant-mAP evaluator for SoccerTrack v2.

Computes per-class temporal mean Average Precision at user-supplied tolerance
windows (1 s and 5 s by default), matching the SoccerNet BAS protocol. Predictions
must use the same JSON schema as ground truth (see docs/format-bas.md).

    python -m src.evaluation.bas_map --pred PRED_ROOT --gt GT_ROOT --matches 128057 132831 \\
        --tolerances 1 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.data_utils.soccertrack_v2 import BAS_LABELS, Event, _parse_bas


def score_many(
    pred_root: Path,
    gt_root: Path,
    match_ids: list[str],
    tolerances_s: Iterable[int] = (1, 5),
) -> dict:
    tolerances_s = list(tolerances_s)
    per_match: dict[str, dict] = {}
    all_tp_fp: dict[int, dict[str, list[tuple[float, int]]]] = {
        t: defaultdict(list) for t in tolerances_s
    }
    all_n_gt: dict[int, dict[str, int]] = {t: defaultdict(int) for t in tolerances_s}

    for mid in match_ids:
        gt_file = gt_root / mid / f"{mid}_12_class_events.json"
        pr_file = pred_root / mid / f"{mid}_12_class_events.json"
        gt = list(_parse_bas(gt_file))
        pred = list(_parse_bas(pr_file))
        per_match[mid] = {}
        for tol in tolerances_s:
            cls_ap = _map_per_class(pred, gt, tol_ms=tol * 1000)
            per_match[mid][f"mAP@{tol}s"] = _mean(cls_ap.values())
            per_match[mid][f"perClass@{tol}s"] = cls_ap
            # Accumulate for dataset-level micro-average (done below)
            for label in BAS_LABELS:
                all_n_gt[tol][label] += sum(1 for e in gt if e.label == label)

    # Dataset-level macro average across matches (simple mean of per-match mAPs)
    overall: dict[str, float] = {}
    for tol in tolerances_s:
        overall[f"mAP@{tol}s"] = _mean(per_match[m][f"mAP@{tol}s"] for m in match_ids)
    return {**per_match, "overall": overall}


def _map_per_class(pred: list[Event], gt: list[Event], tol_ms: int) -> dict[str, float]:
    """Per-class tolerant AP.

    Predictions are ranked by DESCENDING CONFIDENCE, which is what average precision requires.
    This used to sort by ``(half, t_ms)`` -- by time -- which silently discarded any ranking the
    caller had applied, in direct contradiction of the contract stated inside
    ``_ap_tolerant``. The result was not average precision but a time-ordered precision-recall
    traversal: a detector that ranked its output perfectly gained nothing, and one that emitted
    a flood of low-confidence spots was not penalised.

    tests/test_bas_map_identity.py could not catch it, because scoring ground truth against
    itself makes every prediction a true positive and AP is 1.0 in any order. The discriminating
    test is in tests/test_bas_map_ranking.py.

    Ground truth has no scores; ties (including all-None) fall back to time order so the result
    stays deterministic.
    """
    result: dict[str, float] = {}
    for label in BAS_LABELS:
        p = sorted(
            (e for e in pred if e.label == label),
            key=lambda e: (-(e.score if e.score is not None else 0.0), e.half, e.t_ms),
        )
        g = sorted((e for e in gt if e.label == label), key=lambda e: (e.half, e.t_ms))
        result[label] = _ap_tolerant(p, g, tol_ms=tol_ms) if g else float("nan")
    return result


def _ap_tolerant(pred: list[Event], gt: list[Event], tol_ms: int) -> float:
    """Temporal AP with a tolerance window: a prediction is a TP if there is an
    unmatched GT of the same class in the same half within tol_ms."""
    if not gt:
        return float("nan")
    matched = [False] * len(gt)
    tp_fp: list[int] = []
    # `pred` MUST already be ranked best-first; _map_per_class sorts by descending Event.score.
    # Do not re-sort here, and do not sort by time anywhere upstream: average precision is
    # defined over a confidence-ranked list.
    for p in pred:
        best_j = -1
        best_dt = tol_ms + 1
        for j, g in enumerate(gt):
            if matched[j] or g.half != p.half:
                continue
            dt = abs(p.t_ms - g.t_ms)
            if dt <= tol_ms and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            matched[best_j] = True
            tp_fp.append(1)
        else:
            tp_fp.append(0)

    # 11-point interpolated AP
    total_gt = len(gt)
    tp = 0
    precisions: list[tuple[float, float]] = []  # (recall, precision)
    for i, t in enumerate(tp_fp, start=1):
        tp += t
        precisions.append((tp / total_gt, tp / i))
    if not precisions:
        return 0.0
    ap = 0.0
    for r_target in [i / 10 for i in range(11)]:
        p_at_r = max((p for r, p in precisions if r >= r_target), default=0.0)
        ap += p_at_r / 11
    return ap


def _mean(values: Iterable[float]) -> float:
    vs = [v for v in values if v == v]  # drop NaNs
    return sum(vs) / len(vs) if vs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score BAS predictions with tolerant mAP.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--matches", nargs="*", required=True)
    parser.add_argument("--tolerances", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scores = score_many(args.pred, args.gt, args.matches, args.tolerances)
    text = json.dumps(scores, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

"""BAS tolerant-mAP evaluator for SoccerTrack v2.

Per-class temporal average precision at tolerance windows (1 s and 5 s by default),
following the SoccerNet BAS protocol. Predictions use the same JSON schema as ground truth
(see docs/format-bas.md), with an added per-event ``score``.

    python -m src.evaluation.bas_map --pred PRED_ROOT --gt GT_ROOT --matches 128057 132831

THREE THINGS THIS EVALUATOR DOES THAT A NAIVE ONE DOES NOT

1.  IT RANKS BY CONFIDENCE. Average precision is defined over a confidence-ranked list.
    ``_map_per_class`` used to re-sort predictions by ``(half, t_ms)``, silently discarding
    the caller's ranking, so a detector gained nothing from ranking well and was not
    penalised for emitting a flood of low-confidence spots. tests/test_bas_map_ranking.py
    discriminates; the identity test could not, because scoring ground truth against itself
    makes every prediction a true positive and AP is 1.0 in any order.

2.  IT EXCLUDES EVENTS THAT HAVE NO INPUT DATA. Three matches have a third 45-minute period
    with no video and no GSR file; 2,231 events (9.4% of all annotations, 22.8% of test
    match 132831) fall in it. Left in, they are unmatchable ground truth that depresses
    recall by a different amount on every match. See src/data_utils/bas_periods.py.
    tests/test_bas_map_periods.py fails if the filter is removed.

3.  IT MATCHES SOCCERNET EXACTLY, which the paper's citation requires. Their tolerance is a
    HALF-width (``@1s`` means +/-0.5 s), their assignment runs from ground truth to the
    highest-scoring prediction in the window, and their PR curve is sampled at 200 fixed
    confidence thresholds. Getting any of the three wrong changes the number by a factor of
    two or more; tests/test_bas_map_soccernet_parity.py checks all three against a port of
    their source.

4.  IT REPORTS SUPPORT NEXT TO EVERY PER-CLASS NUMBER, and a support-weighted mean beside
    the macro mean. The class imbalance is 301x: Pass has 9,316 events in the benchmark and
    Header has 31, of which 5 are in the test split. A 12-class macro average gives that
    5-instance noise the same weight as Pass. Both means are reported, always labelled, and
    no per-class figure is ever printed without its n.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from src.data_utils.bas_periods import is_evaluable, load_table
from src.data_utils.soccertrack_v2 import BAS_LABELS, Event, _parse_bas


# ---------------------------------------------------------------------------
# Core AP
# ---------------------------------------------------------------------------

def ap_tolerant(scores: Sequence[float], tp_flags: Sequence[int], n_gt: int) -> float:
    """AP exactly as SoccerNet computes it: 200 fixed confidence thresholds, 11-point
    interpolation over the recall axis.

    NOT a cumulative-per-prediction PR curve. SoccerNet sweeps ``np.linspace(0, 1, 200)`` and
    evaluates precision and recall at each threshold, then interpolates. Using every
    prediction as its own threshold gives a finer curve and a different number, and the paper
    cites their protocol, so we match theirs. See tests/test_bas_map_soccernet_parity.py.
    """
    if n_gt == 0:
        return float("nan")
    s = np.asarray(scores, float)
    f = np.asarray(tp_flags, float)
    if f.size == 0:
        return 0.0
    prec, rec = [], []
    for threshold in np.linspace(0, 1, 200):
        idx = np.where(s >= threshold)[0]
        tp = float(f[idx].sum())
        prec.append(tp / len(idx) if len(idx) else 0.0)
        rec.append(tp / n_gt)
    prec, rec = np.array(prec), np.array(rec)
    order = np.argsort(rec)
    prec, rec = prec[order], rec[order]
    ap = 0.0
    for j in np.arange(11) / 10:
        m = rec >= j
        ap += float(prec[m].max()) if m.any() else 0.0
    return float(ap / 11)


def match_greedy(pred: list[Event], gt: list[Event], tol_ms: int) -> np.ndarray:
    """TP/FP flags per prediction, using SoccerNet's assignment.

    TWO THINGS HERE ARE NOT THE OBVIOUS CHOICE, AND BOTH MATTER.

    The window is HALF the nominal tolerance. SoccerNet's condition is
    ``abs(pred - gt) <= delta / 2`` with ``delta = tolerance_seconds * framerate``, so
    "mAP@1s" accepts a prediction within +/-0.5 s. An earlier version of this function
    accepted +/-1 s and therefore scored a strictly easier task than the protocol the paper
    cites, inflating AP by a factor of two to four on random data.

    Assignment runs FROM GROUND TRUTH. For each ground-truth event, in time order, the
    highest-scoring unmatched prediction inside the window is claimed. Matching from
    predictions to the nearest ground truth instead -- the obvious reading of "greedy" --
    disagrees whenever a window holds several candidates, which at a Pass every 2.4 s is the
    normal case rather than an edge case.

    tests/test_bas_map_soccernet_parity.py checks both against a port of their source.
    """
    half = tol_ms / 2.0
    order_p = sorted(range(len(pred)), key=lambda i: (pred[i].half, pred[i].t_ms))
    order_g = sorted(range(len(gt)), key=lambda i: (gt[i].half, gt[i].t_ms))
    flags = np.zeros(len(pred), np.int8)
    used = [False] * len(pred)
    for gi in order_g:
        g = gt[gi]
        best_i, best_score = None, -1.0
        for pi in order_p:
            p = pred[pi]
            if p.half != g.half or used[pi]:
                continue
            if abs(p.t_ms - g.t_ms) > half:
                continue
            sc = p.score if p.score is not None else 0.0
            if sc > best_score:
                best_score, best_i = sc, pi
        if best_i is not None:
            used[best_i] = True
            flags[best_i] = 1
    return flags


def _rank(events: Iterable[Event]) -> list[Event]:
    """Descending confidence; ties fall back to time so the result is deterministic."""
    return sorted(events, key=lambda e: (-(e.score if e.score is not None else 0.0),
                                         e.half, e.t_ms))


def _map_per_class(pred: list[Event], gt: list[Event], tol_ms: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in BAS_LABELS:
        p = _rank(e for e in pred if e.label == label)
        g = sorted((e for e in gt if e.label == label), key=lambda e: (e.half, e.t_ms))
        if not g:
            out[label] = float("nan")
            continue
        flags = match_greedy(p, g, tol_ms)
        out[label] = ap_tolerant([e.score if e.score is not None else 0.0 for e in p],
                                 flags, len(g))
    return out


# ---------------------------------------------------------------------------
# Loading, with the period filter
# ---------------------------------------------------------------------------

def load_events(path: Path, match_periods: Optional[dict]) -> tuple[list[Event], int]:
    """Parse one BAS file, dropping events with no input data. Returns (kept, n_dropped)."""
    raw = list(_parse_bas(path))
    if match_periods is None:
        return raw, 0
    doc = json.loads(path.read_text())
    acts = doc.get("actions", doc.get("annotations"))
    keep = [is_evaluable(a["gameTime"], int(a["position"]), match_periods) for a in acts]
    if len(keep) != len(raw):
        raise RuntimeError(f"{path.name}: parsed {len(raw)} events but the raw file has "
                           f"{len(keep)}; the loader and the period filter disagree")
    kept = [e for e, k in zip(raw, keep) if k]
    return kept, len(raw) - len(kept)


def _mean(values: Iterable[float]) -> float:
    vs = [v for v in values if v == v]
    return sum(vs) / len(vs) if vs else float("nan")


def _weighted(per_class: dict[str, float], support: dict[str, int]) -> float:
    num = sum(per_class[l] * support[l] for l in BAS_LABELS
              if support.get(l) and per_class[l] == per_class[l])
    den = sum(support[l] for l in BAS_LABELS
              if support.get(l) and per_class[l] == per_class[l])
    return num / den if den else float("nan")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_many(
    pred_root: Path,
    gt_root: Path,
    match_ids: list[str],
    tolerances_s: Iterable[int] = (1, 5),
    periods: Optional[dict] = None,
    period_table_path: str | Path | None = None,
    filter_periods: bool = True,
) -> dict:
    """Score predictions against ground truth.

    ``filter_periods`` drops events with no imagery and no tracks from BOTH sides. It
    defaults to True; passing False reproduces the unfiltered behaviour and is used by
    tests/test_bas_map_periods.py to show that the filter changes the answer.

    Returns per-match mAPs plus a dataset-level "overall" block in which detections are
    POOLED across matches before AP is computed -- not an average of per-match mAPs.
    Pooling is what the SoccerNet protocol does, and it is the only form in which a rare
    class is scored on all of its instances at once rather than on two separate handfuls.
    """
    tolerances_s = list(tolerances_s)
    if periods is None and filter_periods:
        periods = load_table(period_table_path)

    per_match: dict[str, dict] = {}
    pooled_pred: dict[str, list[Event]] = defaultdict(list)
    pooled_gt: dict[str, list[Event]] = defaultdict(list)
    dropped = {"gt": 0, "pred": 0}

    for mid in match_ids:
        mp = periods[mid]["periods"] if (periods and filter_periods) else None
        gt, dg = load_events(gt_root / mid / f"{mid}_12_class_events.json", mp)
        pr, dp = load_events(pred_root / mid / f"{mid}_12_class_events.json", mp)
        dropped["gt"] += dg
        dropped["pred"] += dp
        # Keyed by match so pooled matching can never pair a prediction in one match with
        # ground truth in another.
        pooled_gt[mid].extend(gt)
        pooled_pred[mid].extend(pr)

        support = {l: sum(1 for e in gt if e.label == l) for l in BAS_LABELS}
        entry: dict = {"n_gt": len(gt), "n_pred": len(pr), "n_gt_dropped": dg,
                       "n_pred_dropped": dp, "support": support}
        for tol in tolerances_s:
            cls_ap = _map_per_class(pr, gt, tol_ms=tol * 1000)
            entry[f"perClass@{tol}s"] = cls_ap
            entry[f"mAP@{tol}s"] = _mean(cls_ap.values())
            entry[f"mAPw@{tol}s"] = _weighted(cls_ap, support)
        per_match[mid] = entry

    # ---- dataset level: pool detections across matches ---------------------
    support = {l: sum(1 for mid in match_ids for e in pooled_gt[mid] if e.label == l)
               for l in BAS_LABELS}
    overall: dict = {
        "support": support,
        "n_gt": sum(len(pooled_gt[m]) for m in match_ids),
        "n_pred": sum(len(pooled_pred[m]) for m in match_ids),
        "n_gt_dropped": dropped["gt"],
        "n_pred_dropped": dropped["pred"],
    }
    for tol in tolerances_s:
        cls_ap: dict[str, float] = {}
        for label in BAS_LABELS:
            scores: list[float] = []
            flags: list[int] = []
            n_gt = 0
            for mid in match_ids:
                g = sorted((e for e in pooled_gt[mid] if e.label == label),
                           key=lambda e: (e.half, e.t_ms))
                p = _rank(e for e in pooled_pred[mid] if e.label == label)
                n_gt += len(g)
                if not g and not p:
                    continue
                f = match_greedy(p, g, tol * 1000)
                scores.extend(e.score if e.score is not None else 0.0 for e in p)
                flags.extend(f.tolist())
            cls_ap[label] = ap_tolerant(scores, flags, n_gt) if n_gt else float("nan")
        overall[f"perClass@{tol}s"] = cls_ap
        overall[f"mAP@{tol}s"] = _mean(cls_ap.values())
        overall[f"mAPw@{tol}s"] = _weighted(cls_ap, support)
    return {**per_match, "overall": overall}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(scores: dict, match_ids: list[str], tolerances_s=(1, 5)) -> str:
    o = scores["overall"]
    lines = ["PER MATCH"]
    head = f'{"match":10} {"n_gt":>6} {"n_pred":>7}'
    for t in tolerances_s:
        head += f' {"mAP@" + str(t) + "s":>9} {"mAPw@" + str(t) + "s":>10}'
    lines.append(head)
    for m in match_ids:
        e = scores[m]
        row = f'{m:10} {e["n_gt"]:6} {e["n_pred"]:7}'
        for t in tolerances_s:
            row += f' {e[f"mAP@{t}s"]:9.4f} {e[f"mAPw@{t}s"]:10.4f}'
        lines.append(row)
    row = f'{"POOLED":10} {o["n_gt"]:6} {o["n_pred"]:7}'
    for t in tolerances_s:
        row += f' {o[f"mAP@{t}s"]:9.4f} {o[f"mAPw@{t}s"]:10.4f}'
    lines.append(row)
    lines.append("")
    lines.append(f'  events excluded for having no imagery and no tracks: '
                 f'{o["n_gt_dropped"]} ground truth, {o["n_pred_dropped"]} predicted')
    lines.append("")
    lines.append("PER CLASS, POOLED OVER THE SCORED MATCHES  (support = n ground-truth events)")
    head = f'{"class":26} {"support":>8}'
    for t in tolerances_s:
        head += f' {"AP@" + str(t) + "s":>9}'
    lines.append(head)
    for label in BAS_LABELS:
        row = f'{label:26} {o["support"][label]:8}'
        for t in tolerances_s:
            v = o[f"perClass@{t}s"][label]
            row += f' {v:9.4f}' if v == v else f' {"n/a":>9}'
        lines.append(row + ("   <- support below 30; this AP is noise"
                            if o["support"][label] < 30 else ""))
    lines.append("")
    for t in tolerances_s:
        lines.append(f'  macro mAP@{t}s over 12 classes : {o[f"mAP@{t}s"]:.4f}'
                     f'   (equal weight to Header, n={o["support"]["Header"]})')
        lines.append(f'  support-weighted mAP@{t}s      : {o[f"mAPw@{t}s"]:.4f}')
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score BAS predictions with tolerant mAP.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--matches", nargs="*", required=True)
    parser.add_argument("--tolerances", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--periods", default=None,
                        help="period table (default configs/bas_periods.json)")
    parser.add_argument("--no-period-filter", action="store_true",
                        help="score third-period events too; they have no input data")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scores = score_many(args.pred, args.gt, args.matches, args.tolerances,
                        period_table_path=args.periods,
                        filter_periods=not args.no_period_filter)
    print(format_report(scores, args.matches, args.tolerances))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scores, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

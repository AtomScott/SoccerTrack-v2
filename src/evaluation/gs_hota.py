"""GS-HOTA evaluator for SoccerTrack v2 GSR predictions.

Thin wrapper around the reference SoccerNet GSR implementation
(https://github.com/SoccerNet/sn-gamestate). We do **not** reimplement the
metric — we convert SoccerTrack v2's on-disk format into the SoccerNet GSR
expected layout and call the upstream scorer.

    python -m src.evaluation.gs_hota --pred PRED_ROOT --gt GT_ROOT [--matches 128057 132831]

See docs/format-gsr.md for the on-disk annotation layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_many(
    pred_root: Path,
    gt_root: Path,
    match_ids: list[str],
) -> dict:
    """Score `match_ids` with GS-HOTA. Returns `{match_id: metrics, "overall": ...}`.

    Requires `sn-gamestate` to be importable. Raises RuntimeError with install
    guidance otherwise.
    """
    scorer = _load_scorer()
    per_match: dict[str, dict] = {}
    for mid in match_ids:
        per_match[mid] = scorer(pred_root / mid, gt_root / mid)
    overall = _aggregate(per_match)
    return {**per_match, "overall": overall}


def _load_scorer():
    try:
        from sn_gamestate.evaluation import gs_hota as upstream  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "sn-gamestate is not installed. Install the SoccerNet Game State "
            "Reconstruction toolkit (https://github.com/SoccerNet/sn-gamestate) "
            "to use this evaluator."
        ) from e
    return upstream.score_match  # type: ignore[attr-defined]


def _aggregate(per_match: dict[str, dict]) -> dict:
    if not per_match:
        return {}
    keys = set.intersection(*(set(m.keys()) for m in per_match.values()))
    return {k: sum(m[k] for m in per_match.values()) / len(per_match) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score GSR predictions with GS-HOTA.")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction root (mirrors gt layout).")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth gsr/ root.")
    parser.add_argument("--matches", nargs="*", required=True, help="Match IDs to score.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    scores = score_many(args.pred, args.gt, args.matches)
    text = json.dumps(scores, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

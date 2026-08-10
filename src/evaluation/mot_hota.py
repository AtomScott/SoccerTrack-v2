"""MOT evaluator for SoccerTrack v2 — thin wrapper around TrackEval.

Runs HOTA / IDF1 / MOTA on MOTChallenge-format predictions. Ground truth is the
`mot/<match_id>/gt/gt.txt` shipped with SoccerTrack v2; predictions must live under
`<pred_root>/<match_id>/data.txt` in the same format.

    python -m src.evaluation.mot_hota --pred PRED_ROOT --gt GT_ROOT --matches 128057 132831 \\
        --metrics HOTA IDF1 MOTA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_many(
    pred_root: Path,
    gt_root: Path,
    match_ids: list[str],
    metrics: list[str] | None = None,
) -> dict:
    metrics = metrics or ["HOTA", "IDF1", "MOTA"]
    evaluator = _load_trackeval()
    return evaluator(pred_root=pred_root, gt_root=gt_root, match_ids=match_ids, metrics=metrics)


def _load_trackeval():
    try:
        import trackeval  # type: ignore  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "TrackEval is not installed. Install from "
            "https://github.com/JonathonLuiten/TrackEval to use this evaluator."
        ) from e

    def _run(pred_root: Path, gt_root: Path, match_ids: list[str], metrics: list[str]) -> dict:
        # Build TrackEval dataset spec in memory. Kept minimal — if a caller wants to
        # customise (distractor classes, ignore regions, etc.), invoke TrackEval directly.
        import trackeval as te  # type: ignore

        eval_cfg = te.Evaluator.get_default_eval_config()
        eval_cfg["PRINT_CONFIG"] = False
        eval_cfg["TIME_PROGRESS"] = False
        eval_cfg["DISPLAY_LESS_PROGRESS"] = True

        dataset_cfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_cfg.update(
            {
                "GT_FOLDER": str(gt_root),
                "TRACKERS_FOLDER": str(pred_root),
                "TRACKERS_TO_EVAL": None,
                "SPLIT_TO_EVAL": "custom",
                "SEQ_INFO": {mid: None for mid in match_ids},
                "OUTPUT_FOLDER": None,
                "PRINT_CONFIG": False,
            }
        )

        metric_classes = []
        for m in metrics:
            if m == "HOTA":
                metric_classes.append(te.metrics.HOTA())
            elif m == "IDF1":
                metric_classes.append(te.metrics.Identity())
            elif m == "MOTA":
                metric_classes.append(te.metrics.CLEAR())
            else:
                raise ValueError(f"Unknown MOT metric: {m}")

        evaluator = te.Evaluator(eval_cfg)
        results, _ = evaluator.evaluate([te.datasets.MotChallenge2DBox(dataset_cfg)], metric_classes)
        return results

    return _run


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MOT predictions with HOTA / IDF1 / MOTA.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--matches", nargs="*", required=True)
    parser.add_argument("--metrics", nargs="*", default=["HOTA", "IDF1", "MOTA"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scores = score_many(args.pred, args.gt, args.matches, args.metrics)
    text = json.dumps(scores, indent=2, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

"""MOT evaluator for SoccerTrack v2 — thin wrapper around TrackEval.

Runs HOTA / IDF1 / MOTA on MOTChallenge-format predictions via TrackEval's
``MotChallenge2DBox`` dataset class.

Resolved on-disk layout (verified against the installed TrackEval, see below):

* Ground truth:  ``<gt_root>/<seq>/gt/gt.txt``  + ``<gt_root>/<seq>/seqinfo.ini``
* Predictions:   ``<pred_root>/<tracker>/data/<seq>.txt``

where ``<seq>`` is each ``--matches`` entry (a TrackEval sequence name such as
``117099_1st``) and ``<tracker>`` is one prediction-run directory under
``<pred_root>`` (auto-discovered when ``--tracker`` is not given). This is the
native MOTChallenge ``trackers/<tracker>/data/<seq>.txt`` convention.

Why ``SKIP_SPLIT_FOL`` is set
-----------------------------
``MotChallenge2DBox`` defaults to ``SKIP_SPLIT_FOL=False``, which makes it nest
*everything* under a ``BENCHMARK-SPLIT`` folder (here ``MOT17-custom``), i.e. it
would look for GT at ``<gt_root>/MOT17-custom/<seq>/gt/gt.txt`` and trackers at
``<pred_root>/MOT17-custom/<tracker>/data/<seq>.txt``. That contradicts the
un-nested layout documented in docs/format-mot.md and produced by the baseline
writer (baselines/mot/train.py). Setting ``SKIP_SPLIT_FOL=True`` drops the split
folder for both GT and trackers, so the paths above resolve as documented.
(Reproduced against the real TrackEval in the venv: with the default
``SKIP_SPLIT_FOL=False`` the un-nested layout raises
``ini file does not exist: <seq>/seqinfo.ini``; with it True the same data scores.)

    python -m src.evaluation.mot_hota --pred PRED_ROOT --gt GT_ROOT --matches 117099 \\
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
    tracker: str | None = None,
) -> dict:
    """Score MOTChallenge predictions for ``match_ids`` against ground truth.

    Args:
        pred_root: directory containing one tracker run sub-dir, predictions at
            ``<pred_root>/<tracker>/data/<seq>.txt``.
        gt_root: directory containing ``<seq>/gt/gt.txt`` + ``<seq>/seqinfo.ini``.
        match_ids: TrackEval sequence names (e.g. ``["117099_1st", ...]``).
        metrics: subset of ``HOTA`` / ``IDF1`` / ``MOTA`` (default: all three).
        tracker: name of the tracker sub-dir under ``pred_root`` to evaluate. If
            ``None``, TrackEval auto-discovers every sub-dir of ``pred_root``.
    """
    metrics = metrics or ["HOTA", "IDF1", "MOTA"]
    evaluator = _load_trackeval()
    return evaluator(
        pred_root=pred_root,
        gt_root=gt_root,
        match_ids=match_ids,
        metrics=metrics,
        tracker=tracker,
    )


def _load_trackeval():
    try:
        import trackeval  # type: ignore  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "TrackEval is not installed. Install from "
            "https://github.com/JonathonLuiten/TrackEval to use this evaluator."
        ) from e

    def _run(
        pred_root: Path,
        gt_root: Path,
        match_ids: list[str],
        metrics: list[str],
        tracker: str | None = None,
    ) -> dict:
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
                # Auto-discover the tracker sub-dir(s) of pred_root unless one is named.
                "TRACKERS_TO_EVAL": None if tracker is None else [tracker],
                "SPLIT_TO_EVAL": "custom",
                "SEQ_INFO": {mid: None for mid in match_ids},
                "OUTPUT_FOLDER": None,
                "PRINT_CONFIG": False,
                # Drop the implicit 'BENCHMARK-SPLIT' (MOT17-custom) folder so GT and
                # tracker paths resolve un-nested, exactly as docs/format-mot.md and
                # baselines/mot/train.py lay them out:
                #   GT:    <gt_root>/<seq>/gt/gt.txt   (+ <gt_root>/<seq>/seqinfo.ini)
                #   preds: <pred_root>/<tracker>/data/<seq>.txt
                # Without this, MotChallenge2DBox prepends MOT17-custom/ to both and the
                # documented layout fails to score (verified against the installed
                # TrackEval — see module docstring).
                "SKIP_SPLIT_FOL": True,
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
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Prediction root; predictions at <pred>/<tracker>/data/<seq>.txt.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Ground-truth root; GT at <gt>/<seq>/gt/gt.txt (+ <seq>/seqinfo.ini).",
    )
    parser.add_argument("--matches", nargs="*", required=True)
    parser.add_argument("--metrics", nargs="*", default=["HOTA", "IDF1", "MOTA"])
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Tracker sub-dir of --pred to evaluate (default: auto-discover all).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scores = score_many(args.pred, args.gt, args.matches, args.metrics, tracker=args.tracker)
    text = json.dumps(scores, indent=2, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

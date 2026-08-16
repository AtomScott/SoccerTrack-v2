"""Run the prior rule-based detector and convert its output to the benchmark schema.

WHAT THIS IS
    scripts/event_detection_tracking/event_detection.py is the rule-based event detector that
    predates this work: it derives possession from ball-to-player distances and emits events
    from possession transitions and ball geometry. Running it gives the paper a prior method
    to compare the learned baseline against, on identical data and an identical metric.

FOUR THINGS THAT HAD TO BE FIXED OR STATED BEFORE IT COULD BE COMPARED

  1. ITS INPUT DID NOT EXIST. It reads a per-match pitch-plane CSV from a directory that is
     empty for all ten matches. scripts/bas/make_pitch_plane_csv.py regenerates it from the
     raw tracking XML, verified byte-identical against the two surviving originals.

  2. IT COULD NOT RUN AS SHIPPED. It selected the frame at `match_time == 0.0` exactly; no
     match's clock lands on zero (first frames are at -33, 40 or 520 ms, advancing in 40 ms
     steps), so the selection was empty and it raised KeyError on every match. A second latent
     bug left `plus_team_id` unbound whenever the first team's centroid was in the far half.
     Both are repaired in place, with comments, since the file was otherwise dead code.

  3. IT EXPECTS METRES, NOT THE NORMALISED UNITS THE SURVIVING CSVs USE. Its own comparison
     `centroid_x < 52.5` gives that away. The CSVs fed to it are generated with
     `--units metres`.

  4. IT EMITS NO CONFIDENCE. Every prediction carries the literal string "0.5", so its output
     has no ranking at all. Average precision is defined over a confidence-ranked list, so a
     constant-confidence detector cannot be scored on the ranking half of the metric, and its
     AP is bounded by what a single unordered set can achieve. This is a property of the
     method, not of the evaluation, and it must be stated wherever its numbers appear rather
     than quietly ignored.

    It also emits a `GOAL KICK` label, which is not one of the twelve benchmark classes. Those
    predictions are dropped and counted.

    The regenerated input is UNFILTERED, while the original filename says "filtered" and the
    repository history mentions a Kalman filter. Its thresholds were presumably tuned against
    the smoothed version, so these scores are a lower bound on what it achieved originally.

USAGE
    python scripts/bas/run_prior_detector.py --convert --matches 128057 132831
    python scripts/bas/run_prior_detector.py --convert --all --score
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_utils.soccertrack_v2 import BAS_LABELS  # noqa: E402

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
_CANON = {l.casefold(): l for l in BAS_LABELS}


def convert(raw_dir: Path, out_root: Path, match: str, periods: dict | None = None) -> dict:
    """Convert one match, correcting the clock the detector reports on.

    THE DETECTOR REPORTS TRACKING TIME, THE BENCHMARK USES EVENT TIME. It copies `match_time`
    straight out of the tracking CSV, but the BAS annotations live on a clock that differs
    from the tracking clock by one second on fourteen of the twenty halves -- the offset this
    work measured (docs/bas-findings.md section 3). Uncorrected, the prior method scores
    0.004 macro mAP@1s on 128057; corrected, 0.089. Reporting the uncorrected figure would
    have made a clock mismatch look like a method failure.

    The correction is the per-half `t0_event_shift_frames` from configs/bas_periods.json,
    which was measured against the ball, not fitted to any score.
    """
    src = raw_dir / match / f"{match}_12_class_events_detection.json"
    if not src.exists():
        return {"match": match, "ok": False, "reason": "detector output not found"}
    doc = json.loads(src.read_text())
    preds = doc.get("predictions", doc.get("actions", doc.get("annotations", [])))
    kept, dropped = [], Counter()
    for p in preds:
        lab = _CANON.get(str(p["label"]).casefold())
        if lab is None:
            dropped[str(p["label"])] += 1
            continue
        t = int(float(p["position"]))
        kept.append({
            "gameTime": p.get("gameTime", ""),
            "label": lab,
            "position": str(t),
            "team": p.get("team") or None,
            # Constant by construction -- the detector has no confidence. Preserved rather
            # than replaced with a fabricated ranking.
            "score": float(p.get("confidence", 0.5)),
        })
    # gameTime from the detector is "<period> - <m:ss>" with a single-digit minute; rebuild it
    # in the benchmark's own form so the period prefix parses.
    n_shifted = 0
    for a in kept:
        t = int(a["position"])
        period = 1 if t < 2_700_000 else 2
        if periods is not None:
            shift = periods[str(period)].get("t0_event_shift_frames")
            if shift:
                t -= int(shift) * 40
                a["position"] = str(t)
                n_shifted += 1
        s = max(0, t) // 1000
        a["gameTime"] = f"{period} - {s // 60:02d}:{s % 60:02d}"
    d = out_root / match
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{match}_12_class_events.json").write_text(
        json.dumps({"match_id": match, "fps": 25.0, "actions": kept}, indent=1))
    return {"match": match, "ok": True, "kept": len(kept), "dropped": dict(dropped),
            "clock_corrected": n_shifted}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="run_rule/data/interim/event_detection_tracking")
    ap.add_argument("--out", default="outputs/prior/pred")
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--gt", default="/data/share/SoccerTrack-v2/data/production/bas")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--no-clock-correction", action="store_true",
                    help="report on the tracking clock, as the detector emits it")
    a = ap.parse_args()

    matches = MATCHES if a.all else (a.matches or [])
    if not matches:
        ap.error("give --all or --matches")
    out = Path(a.out)

    if a.convert:
        table = None
        if not a.no_clock_correction:
            table = json.loads(Path(a.periods).read_text())["matches"]
        total_dropped = Counter()
        done = []
        for m in matches:
            r = convert(Path(a.raw), out, m, table[m]["periods"] if table else None)
            if not r["ok"]:
                print(f"  {m}: {r['reason']}")
                continue
            done.append(m)
            total_dropped.update(r["dropped"])
            print(f'  {m}: {r["kept"]:,} predictions'
                  + (f', {r["clock_corrected"]:,} clock-corrected' if r["clock_corrected"] else '')
                  + (f'  (dropped {r["dropped"]})' if r["dropped"] else ''))
        if total_dropped:
            print(f"\n  labels outside the twelve-class set, dropped: {dict(total_dropped)}")
        matches = done

    if a.score and matches:
        from src.evaluation.bas_map import format_report, score_many
        sc = score_many(out, Path(a.gt), matches)
        print()
        print(format_report(sc, matches))
        print()
        print("  NOTE: every prediction carries the same confidence, so average precision is")
        print("  measured on an unranked list. This bounds what the method can score and is a")
        print("  property of the detector, not of the metric.")
        (out.parent / "scores.json").write_text(json.dumps(sc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

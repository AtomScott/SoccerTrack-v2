"""Chance baselines for BAS, so the trajectory numbers can be read.

WHY THIS EXISTS
    "mAP@1s = 0.16" means nothing on its own, and it means less than nothing on this
    dataset, where two of the twelve classes account for 82% of all events. Pass occurs
    every 2.4 s of match time and Drive every 2.7 s. A model that emits Pass at a fixed
    cadence and nothing else will land a lot of true positives inside a 1 s tolerance
    window purely because the class is dense. Any reported trajectory result has to be read
    against that floor, not against zero.

    Two floors are provided, both using ONLY statistics that a system is entitled to know
    from the training split -- never the test events themselves:

    uniform  Per class, spots evenly spaced at the class's mean training-set rate, filling
             each half. This is the strongest baseline that knows nothing about the match:
             it exploits event density and nothing else.

    random   Per class, the same NUMBER of spots as `uniform`, placed at uniformly random
             times with random scores. This separates "spacing events evenly is a good
             strategy" from "guessing the rate is enough".

    A third, `oracle-rate`, uses the TEST split's own per-class counts to set the number of
    spots. It is not a legitimate baseline -- it peeks -- and is reported only to show how
    much of any score is attributable to knowing the class frequencies.

USAGE
    python scripts/bas/baseline_priors.py --kind uniform --out outputs/bas/pred_uniform
    python scripts/bas/baseline_priors.py --kind random  --out outputs/bas/pred_random
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_utils.bas_periods import is_evaluable, load_table  # noqa: E402

BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
_CANON = {l.casefold(): l for l in BAS_LABELS}

TEST = ["128057", "132831"]
TRAIN = ["117092", "117093", "118575", "118576", "118577", "118578", "128058", "132877"]


def read_evaluable(data: Path, match: str, table: dict) -> list[tuple[int, int, str]]:
    """(period, position_ms, label) for events with input data."""
    path = data / "production" / "bas" / match / f"{match}_12_class_events.json"
    acts = json.loads(path.read_text())
    acts = acts.get("actions", acts.get("annotations"))
    mp = table[match]["periods"]
    out = []
    for a in acts:
        if not is_evaluable(a["gameTime"], int(a["position"]), mp):
            continue
        prefix = int(str(a["gameTime"]).split(" - ", 1)[0])
        out.append((prefix, int(a["position"]), _CANON[str(a["label"]).casefold()]))
    return out


def half_span_ms(table: dict, match: str, period: int) -> tuple[float, float]:
    spec = table[match]["periods"][str(period)]
    return spec["t0_ms"], spec["t0_ms"] + (spec["n_frames"] - 1) * 40.0


def train_rates(data: Path, table: dict) -> dict[str, float]:
    """Mean events per second of tracked play, per class, over the training matches only."""
    counts = {l: 0 for l in BAS_LABELS}
    seconds = 0.0
    for m in TRAIN:
        for p in (1, 2):
            lo, hi = half_span_ms(table, m, p)
            seconds += (hi - lo) / 1000.0
        for _p, _t, lab in read_evaluable(data, m, table):
            counts[lab] += 1
    return {l: counts[l] / seconds for l in BAS_LABELS}


def ms_to_clock(t_ms: int) -> str:
    s = max(0, int(t_ms)) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def build(kind: str, data: Path, table: dict, matches: list[str], out_root: Path,
          seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    rates = train_rates(data, table)
    written = {}
    for m in matches:
        actions = []
        for p in (1, 2):
            lo, hi = half_span_ms(table, m, p)
            dur_s = (hi - lo) / 1000.0
            for label in BAS_LABELS:
                if kind == "oracle-rate":
                    n = sum(1 for q, _t, l in read_evaluable(data, m, table)
                            if q == p and l == label)
                else:
                    n = int(round(rates[label] * dur_s))
                if n <= 0:
                    continue
                if kind == "random":
                    ts = np.sort(rng.uniform(lo, hi, n))
                    sc = rng.uniform(0.0, 1.0, n)
                else:
                    # evenly spaced, half a gap in from each end
                    ts = lo + (np.arange(n) + 0.5) * (hi - lo) / n
                    # A flat score would make the ranking arbitrary; a smooth ramp keeps it
                    # deterministic without encoding any information about the match.
                    sc = np.linspace(0.9, 0.1, n)
                for t, s in zip(ts, sc):
                    actions.append({"gameTime": f"{p} - {ms_to_clock(t)}", "label": label,
                                    "position": str(int(round(t))), "team": "left",
                                    "score": round(float(s), 6)})
        actions.sort(key=lambda a: -a["score"])
        d = out_root / m
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{m}_12_class_events.json").write_text(
            json.dumps({"match_id": m, "fps": 25.0, "actions": actions}, indent=1))
        written[m] = len(actions)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--kind", choices=["uniform", "random", "oracle-rate"], default="uniform")
    ap.add_argument("--matches", nargs="*", default=TEST)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = Path(a.data)
    table = load_table(a.periods)
    rates = train_rates(data, table)
    print("per-class rate estimated on the TRAINING matches (events per second of play):")
    for l in BAS_LABELS:
        gap = 1.0 / rates[l] if rates[l] > 0 else float("inf")
        print(f"  {l:26} {rates[l]:.5f}/s   one every {gap:7.1f} s")
    n = build(a.kind, data, table, a.matches, Path(a.out), a.seed)
    print(f"\n{a.kind} baseline: " + ", ".join(f"{k} {v:,} spots" for k, v in n.items())
          + f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

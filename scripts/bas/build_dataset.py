"""Pair per-frame trajectory features with BAS event targets, one file per half.

Reads the streamed track caches (scripts/bas/extract_tracks.py) and the released BAS event
files, and writes a training-ready .npz per half:

    feat    (T, 82)  float32   see src/bas/features.py
    target  (T, 12)  float32   per-class soft target, peaked at the event
    frames  (T,)     int32     1-based GSR frame index of each row
    ev_frame, ev_class, ev_t_ms                                the events themselves

PERIOD HANDLING
    Only periods 1 and 2 are built, because only they have tracks. The period of an event,
    and its frame index, come from configs/bas_periods.json via the rule established in
    scripts/bas/audit_annotations.py: an event belongs to a period when its frame lands
    inside that period's annotated GSR frame range. Third-period events -- 2,232 of them,
    9.4% of the annotations -- are dropped here, and the count dropped is reported so the
    number never becomes invisible.

THE TARGET
    A spotting target must be sharp enough to localise and smooth enough to train. Each
    event contributes a Gaussian of sigma 1.5 rows (300 ms at the default 5 Hz), truncated
    at +/-3 rows, taking the maximum where events overlap. Classes are INDEPENDENT sigmoid
    targets rather than a softmax over classes plus background, because simultaneous events
    are real and documented: a goal is annotated as a Shot and a Goal at the same position
    (docs/format-bas.md, "Dual actions at the same timestamp. ... Do not collapse.").
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.bas.features import build_features, N_FEATURES  # noqa: E402

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
HALF_NAME = {1: "1st", 2: "2nd"}
BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
_CANON = {l.casefold(): l for l in BAS_LABELS}
_INDEX = {l: i for i, l in enumerate(BAS_LABELS)}

TARGET_SIGMA_ROWS = 1.5
TARGET_TRUNC_ROWS = 3


def load_events(data: Path, match: str, periods: dict) -> list[tuple[int, int, int]]:
    """(period, frame, class_index, t_ms) for every event that has tracks."""
    path = data / "production" / "bas" / match / f"{match}_12_class_events.json"
    doc = json.loads(path.read_text())
    acts = doc.get("actions", doc.get("annotations"))
    kept, dropped = [], 0
    for a in acts:
        gt = str(a["gameTime"])
        pos = int(a["position"])
        prefix = int(gt.split(" - ", 1)[0]) if " - " in gt else None
        placed = False
        if prefix in (1, 2):
            spec = periods[str(prefix)]
            if spec["n_frames"] is not None:
                frame = int(round((pos - spec["t0_ms"]) / 40.0)) + 1
                if 1 <= frame <= spec["n_frames"]:
                    kept.append((prefix, frame, _INDEX[_CANON[str(a["label"]).casefold()]], pos))
                    placed = True
        if not placed:
            dropped += 1
    return kept, dropped


def build_half(args) -> dict:
    data, tracks, out_dir, match, half, periods, stride = args
    data, tracks, out_dir = Path(data), Path(tracks), Path(out_dir)
    npz = tracks / f"{match}_{HALF_NAME[half]}_tracks.npz"
    if not npz.exists():
        return {"match": match, "half": half, "ok": False, "reason": "no track cache"}

    feat, frames = build_features(np.load(npz), stride=stride)
    events, dropped = load_events(data, match, periods)
    ev = [(f, c, t) for p, f, c, t in events if p == half]

    T = feat.shape[0]
    target = np.zeros((T, len(BAS_LABELS)), np.float32)
    # sampled row index of a frame: rows are frames 1, 1+stride, 1+2*stride, ...
    for f, c, _t in ev:
        r = int(round((f - 1) / stride))
        lo, hi = max(0, r - TARGET_TRUNC_ROWS), min(T, r + TARGET_TRUNC_ROWS + 1)
        if lo >= hi:
            continue
        d = np.arange(lo, hi) - r
        g = np.exp(-0.5 * (d / TARGET_SIGMA_ROWS) ** 2).astype(np.float32)
        np.maximum(target[lo:hi, c], g, out=target[lo:hi, c])

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{match}_{HALF_NAME[half]}_dataset.npz"
    np.savez_compressed(
        dst, feat=feat, target=target, frames=frames,
        ev_frame=np.array([e[0] for e in ev], np.int32),
        ev_class=np.array([e[1] for e in ev], np.int8),
        ev_t_ms=np.array([e[2] for e in ev], np.int64),
        stride=np.int32(stride),
    )
    return {"match": match, "half": half, "ok": True, "reason": "", "T": T,
            "events": len(ev), "dropped_match_total": dropped, "out": str(dst)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--tracks", default="/data/share/SoccerTrack-v2/data/derived/bas/tracks")
    ap.add_argument("--out", default="/data/share/SoccerTrack-v2/data/derived/bas/dataset")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--stride", type=int, default=5, help="frame subsampling; 5 = 5 Hz")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    table = json.loads(Path(a.periods).read_text())["matches"]
    jobs = [(a.data, a.tracks, a.out, m, h, table[m]["periods"], a.stride)
            for m in MATCHES for h in (1, 2)]
    print(f"building {len(jobs)} halves at stride {a.stride} "
          f"({25/a.stride:.0f} Hz), {N_FEATURES} features -> {a.out}\n")
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed({ex.submit(build_half, j): j for j in jobs}):
            r = fu.result()
            results.append(r)
            if r["ok"]:
                print(f"  {r['match']} {HALF_NAME[r['half']]}: T={r['T']:,} rows, "
                      f"{r['events']:,} events", flush=True)
            else:
                print(f"  FAILED {r['match']} {r['half']}: {r['reason']}")
    ok = [r for r in results if r["ok"]]
    total_ev = sum(r["events"] for r in ok)
    dropped = sum(r["dropped_match_total"] for r in ok) // 2  # counted once per half
    print(f"\n{len(ok)}/{len(jobs)} halves built; {total_ev:,} events with tracks, "
          f"{dropped:,} annotated events dropped for having none (third period)")
    return 0 if len(ok) == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())

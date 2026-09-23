"""Measure, per half, the match-clock time of GSR GameState frame 1.

WHY THIS EXISTS
    Everything downstream needs `t0_ms`: an event's frame index is
    `1 + (position - t0_ms)/40`, so an error in t0 attaches every event in that half to the
    wrong player configuration. This is the BAS analogue of scripts/gsr/measure_frame_offset.py
    and it exists for the same reason: the first version of this table COMPUTED t0 where it
    could not read it, and was wrong by 1.0-1.5 SECONDS on two of the ten matches -- one of
    them in the test split.

WHAT WENT WRONG THE FIRST TIME, so it is not repeated
    t0 was derived from `<period frameStart= matchTimeStart=>` in
    <match>_tracker_box_metadata.xml. That works for eight matches. 132831 and 132877 carry
    NO <period> elements at all, and for those two t0 fell back to the nominal 0 and
    2,700,000 ms. Measured, the true values are 1,040 / 2,701,000 and 1,520 / 2,701,000 --
    the fallback was out by 1.04, 1.00, 1.52 and 1.00 seconds.

    Two checks failed to catch it:
      * the throw-in touchline check in audit_annotations.py passes at z = 13-21 for both
        matches even when displaced by a second, because a thrower stands on the line for
        several seconds either side of the throw. It was built to catch a gross error and
        it did its job; it was never sharp enough for this.
      * comparing against the interim pitch-plane CSVs covered only 117092 and 117093,
        neither of which is affected.

HOW IT IS MEASURED NOW
    <match>_tracker_box_data.xml carries, for EVERY frame of every period, a frameNumber, a
    matchTime, and the position of every player. The released GSR annotations are derived
    from it. So the GSR-to-raw frame offset can be measured directly by asking which
    alignment makes the SAME player_id's position agree between the two sources -- and at
    the correct alignment they are the same data, so the residual must be ZERO, not merely
    small. That is a much stronger acceptance test than "this looks about right".

    Positions are quantised to 1.05 m, and a player moving at the median 1.4 m/s does not
    cross one quantisation step in ten frames, so a median over all players is FLAT across a
    ten-frame plateau and cannot locate the offset. The statistic is therefore the MEAN error
    over players the provider reports as moving faster than 4 m/s, which shift ~0.16 m per
    frame and resolve the offset to a single frame. Every half produces a clean V: exactly
    0.0000 m at the optimum and 0.21-0.29 m one frame either side.

    132831 and 132877 also need their y inverted relative to the raw file; the released GSR
    already applies that, so the flip is detected here and reported rather than corrected.

USAGE
    python scripts/bas/measure_t0.py --all
    python scripts/bas/measure_t0.py --all --write configs/bas_periods.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
HALF_NAME = {1: "1st", 2: "2nd"}
PERIOD_TAG = {1: b"FIRST_HALF", 2: b"SECOND_HALF"}
PITCH_L, PITCH_W = 105.0, 68.0
FRAME_MS = 40.0

_FRAME = re.compile(
    rb'<frame matchTime="(-?\d+)" frameNumber="(\d+)" eventPeriod="(\w+)" '
    rb'ballStatus="(\w+)">(.*?)</frame>', re.S)
_PLAYER = re.compile(rb'<player playerId="(\d+)" loc="\[([^,]+), ([^\]]+)\]" speed="([^"]*)"/>')

FAST_MS = 4.0          # provider speed threshold, m/s
SEARCH_HALF_WIDTH = 40  # frames either side of (first raw frame - 1)
ACCEPT_RESIDUAL_M = 0.01


def _gsr_grid(tracks: Path, match: str, half: int):
    d = np.load(tracks / f"{match}_{HALF_NAME[half]}_tracks.npz")
    n_f = int(d["frame"].max())
    pl = np.unique(d["player_id"])
    col = {int(p): i for i, p in enumerate(pl.tolist())}
    X = np.full((n_f + 2, pl.size), np.nan, np.float32)
    Y = np.full_like(X, np.nan)
    idx = [col[int(p)] for p in d["player_id"].tolist()]
    X[d["frame"], idx] = d["x"]
    Y[d["frame"], idx] = d["y"]
    return X, Y, col, n_f


def measure_half(raw_xml: bytes, tracks: Path, match: str, half: int,
                 max_samples: int = 300) -> dict:
    X, Y, col, n_f = _gsr_grid(tracks, match, half)
    tag = PERIOD_TAG[half]
    first_frame = first_mt = None
    samples: list[tuple[int, dict]] = []
    seen = 0
    for g in _FRAME.finditer(raw_xml):
        if g.group(3) != tag:
            continue
        if first_frame is None:
            first_frame, first_mt = int(g.group(2)), int(g.group(1))
        seen += 1
        if seen % 250 == 0 and len(samples) < max_samples:
            fast = {}
            for q in _PLAYER.finditer(g.group(5)):
                try:
                    if float(q.group(4)) > FAST_MS:
                        fast[int(q.group(1))] = (float(q.group(2)), float(q.group(3)))
                except ValueError:
                    continue
            if fast:
                samples.append((int(g.group(2)), fast))
    if first_frame is None or not samples:
        return {"match": match, "half": half, "ok": False, "reason": "no raw frames"}

    base = first_frame - 1
    best = None
    for flip in (False, True):
        for off in range(base - SEARCH_HALF_WIDTH, base + SEARCH_HALF_WIDTH + 1):
            errs = []
            for fn, fast in samples:
                gs = fn - off
                if not (1 <= gs <= n_f):
                    continue
                for pid, (bx, by) in fast.items():
                    j = col.get(pid)
                    if j is None or np.isnan(X[gs, j]):
                        continue
                    ry = (0.5 - by if flip else by - 0.5) * PITCH_W
                    errs.append(abs(X[gs, j] - (bx - 0.5) * PITCH_L) + abs(Y[gs, j] - ry))
            if len(errs) < 200:
                continue
            e = float(np.mean(errs))
            if best is None or e < best[0]:
                best = (e, off, flip, len(errs))
    if best is None:
        return {"match": match, "half": half, "ok": False, "reason": "too few comparisons"}

    err, off, flip, n_cmp = best
    # Neighbour errors, to prove the optimum is a sharp V and not a plateau.
    def at(o):
        errs = []
        for fn, fast in samples:
            gs = fn - o
            if not (1 <= gs <= n_f):
                continue
            for pid, (bx, by) in fast.items():
                j = col.get(pid)
                if j is None or np.isnan(X[gs, j]):
                    continue
                ry = (0.5 - by if flip else by - 0.5) * PITCH_W
                errs.append(abs(X[gs, j] - (bx - 0.5) * PITCH_L) + abs(Y[gs, j] - ry))
        return float(np.mean(errs)) if len(errs) >= 200 else float("nan")

    left, right = at(off - 1), at(off + 1)
    t0 = first_mt - (first_frame - off - 1) * FRAME_MS
    sharp = (err <= ACCEPT_RESIDUAL_M
             and not (np.isnan(left) or np.isnan(right))
             and min(left, right) > err + 0.05)
    return {"match": match, "half": half, "ok": bool(sharp),
            "reason": "" if sharp else "residual not zero, or optimum not sharp",
            "raw_first_frame": first_frame, "raw_first_match_time": first_mt,
            "gsr_to_raw_offset": off, "y_flip_vs_raw": flip, "t0_ms": float(t0),
            "residual_m": round(err, 5), "residual_left": round(left, 4),
            "residual_right": round(right, 4), "n_compared": n_cmp,
            "n_gsr_frames": n_f}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--tracks", default="/data/share/SoccerTrack-v2/data/derived/bas/tracks")
    ap.add_argument("--match")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--write", nargs="?", const="configs/bas_periods.json", default=None,
                    help="update t0_ms in this period table from the measurement")
    a = ap.parse_args()
    data, tracks = Path(a.data), Path(a.tracks)
    matches = MATCHES if a.all else ([a.match] if a.match else [])
    if not matches:
        ap.error("give --all or --match")

    print(f'{"match":8} {"half":5} {"rawF0":>7} {"mt@F0":>9} {"offset":>7} {"yflip":>6} '
          f'{"resid":>7} {"-1":>6} {"+1":>6} {"n":>6}  {"t0_ms":>10}  verdict')
    rows = []
    for m in matches:
        xml = (data / "raw" / m / f"{m}_tracker_box_data.xml").read_bytes()
        for half in (1, 2):
            r = measure_half(xml, tracks, m, half)
            rows.append(r)
            if not r["ok"] and "raw_first_frame" not in r:
                print(f'{m:8} {HALF_NAME[half]:5} FAILED: {r["reason"]}')
                continue
            print(f'{m:8} {HALF_NAME[half]:5} {r["raw_first_frame"]:7} '
                  f'{r["raw_first_match_time"]:9} {r["gsr_to_raw_offset"]:7} '
                  f'{str(r["y_flip_vs_raw"]):>6} {r["residual_m"]:7.4f} '
                  f'{r["residual_left"]:6.3f} {r["residual_right"]:6.3f} {r["n_compared"]:6} '
                  f'{r["t0_ms"]:10.0f}  {"OK" if r["ok"] else "NOT SHARP"}', flush=True)
        del xml

    bad = [r for r in rows if not r["ok"]]
    print(f'\n{len(rows) - len(bad)}/{len(rows)} halves measured to a zero residual with a '
          f'sharp optimum')
    for r in bad:
        print(f'  SUSPECT {r["match"]} {HALF_NAME[r["half"]]}: {r["reason"]}')

    if a.write:
        p = Path(a.write)
        if p.exists():
            table = json.loads(p.read_text())
        else:
            # measure_t0 OWNS the period table. Bootstrapping it here rather than in
            # audit_annotations.py removes a two-step in which the audit recomputed t0 from
            # metadata and silently overwrote the measured value.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from scripts.bas.audit_annotations import build_period_shell
            table = build_period_shell(data)
        changed = []
        for r in rows:
            if not r["ok"]:
                continue
            spec = table["matches"][r["match"]]["periods"][str(r["half"])]
            if abs(spec["t0_ms"] - r["t0_ms"]) > 1e-6:
                changed.append((r["match"], r["half"], spec["t0_ms"], r["t0_ms"]))
            spec["t0_ms"] = r["t0_ms"]
            spec["t0_source"] = "measured_vs_tracker_box_data"
            # n_frames must be the last frame that actually CARRIES an annotation, not the
            # declared info.seq_length. 132831 and 132877 declare 25 frames more than they
            # annotate, and events landing in that gap passed the period filter while having
            # no features to score against -- three events across the dataset.
            if spec.get("n_frames") != r["n_gsr_frames"]:
                spec["n_frames_declared_seq_length"] = spec.get("n_frames")
            spec["n_frames"] = r["n_gsr_frames"]
            spec["t0_residual_m"] = r["residual_m"]
            spec["gsr_to_raw_offset"] = r["gsr_to_raw_offset"]
            spec["y_flip_vs_raw"] = r["y_flip_vs_raw"]
        table["_comment"] = (
            "t0_ms and n_frames MEASURED by scripts/bas/measure_t0.py against "
            "<match>_tracker_box_data.xml, to a zero residual with a sharp optimum. "
            "An event's frame index is 1 + (position - t0_ms)/40. Period 3 has no imagery "
            "and no tracks in any match and is outside the benchmark.")
        p.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\nwrote {p}")
        if changed:
            print("t0 CHANGED for:")
            for m, h, old, new in changed:
                print(f"  {m} {HALF_NAME[h]}: {old:.0f} -> {new:.0f} ms "
                      f"({(new-old)/1000:+.3f} s)")
        else:
            print("no t0 changed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

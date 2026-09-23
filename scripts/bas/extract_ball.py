"""Extract the ball track per half from the raw provider tracking data.

WHY THIS EXISTS
    The released GSR annotations contain no ball at all -- 20 outfield players and 2
    goalkeepers, and nothing else. But <match>_tracker_box_data.xml carries a ball position
    on EVERY frame of ALL TEN matches, with zero missing values, alongside a per-frame
    ballStatus. Atom's decision (2026-08-14) is to ship the ball with the dataset and report
    both a with-ball and a without-ball benchmark.

    The ball is also the only sharp instrument available for measuring the BAS-to-GSR time
    alignment. A struck ball accelerates within a frame or two, so its speed steps at the
    true event time; the throw-in touchline check used before is flat over several seconds
    and cannot resolve a one-second error. See scripts/bas/measure_event_offset.py.

CONVENTIONS, BOTH MEASURED
    Frame indexing. The raw file numbers frames per period starting at 251 for eight matches
    and at 1 for 132831 and 132877, and the GSR-to-raw offset differs accordingly. It is read
    from configs/bas_periods.json, where scripts/bas/measure_t0.py measured it to a zero
    residual against the players.

    The y axis. The released GSR applies a per-match y inversion (132831 and 132877 are
    inverted relative to the raw file; the other eight are not). The ball has to end up in
    the SAME frame of reference as those player positions or nothing downstream is
    comparable, so --check re-derives the ball's convention from scratch by asking which
    choice puts the ball at somebody's feet, and reports it per match rather than assuming
    the players' answer transfers. In 132877 the raw ball is known to disagree with its own
    players, so it cannot be assumed.

BALLSTATUS IS DELIBERATELY KEPT SEPARATE AND FLAGGED
    On eight matches ballStatus takes BALLOUT / HOME / AWAY / NEUTRAL. BALLOUT is close to a
    direct label for the Out class, so any model consuming it is partly reading the answer.
    On 132831 and 132877 it is the constant INPLAY and carries no information whatsoever --
    and 132831 is in the test split, so a detector tuned on the other eight would collapse
    there. It is extracted so the leak can be quantified, and must not be used as a feature.

USAGE
    python scripts/bas/extract_ball.py --all
    python scripts/bas/extract_ball.py --check
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

_FRAME = re.compile(
    rb'<frame matchTime="(-?\d+)" frameNumber="(\d+)" eventPeriod="(\w+)" '
    rb'ballStatus="(\w+)">(.*?)</frame>', re.S)
_BALL = re.compile(rb'<ball playerId="ball" loc="\[([^,]+), ([^\]]+)\]" speed="([^"]*)"/>')
_PLAYER = re.compile(rb'<player playerId="(\d+)" loc="\[([^,]+), ([^\]]+)\]"')

STATUS_CODE = {"BALLOUT": 0, "HOME": 1, "AWAY": 2, "NEUTRAL": 3, "INPLAY": 4}


def _raw_rows(xml: bytes, half: int):
    """Yield (frameNumber, matchTime, ballStatus, ball_xy_or_None, players dict)."""
    tag = PERIOD_TAG[half]
    for g in _FRAME.finditer(xml):
        if g.group(3) != tag:
            continue
        body = g.group(5)
        b = _BALL.search(body)
        xy = None
        if b is not None:
            try:
                xy = (float(b.group(1)), float(b.group(2)))
            except ValueError:
                xy = None
        yield int(g.group(2)), int(g.group(1)), g.group(4).decode(), xy, body


def extract_half(xml: bytes, match: str, half: int, spec: dict, out_dir: Path,
                 flip_y: bool) -> dict:
    off = spec["gsr_to_raw_offset"]
    n_frames = spec["n_frames"]
    frame = np.zeros(n_frames + 1, np.int32)
    bx = np.full(n_frames + 1, np.nan, np.float32)
    by = np.full(n_frames + 1, np.nan, np.float32)
    status = np.full(n_frames + 1, -1, np.int8)
    seen = 0
    for fn, _mt, st, xy, _body in _raw_rows(xml, half):
        gs = fn - off
        if not (1 <= gs <= n_frames):
            continue
        frame[gs] = gs
        status[gs] = STATUS_CODE.get(st, -1)
        if xy is not None:
            bx[gs] = (xy[0] - 0.5) * PITCH_L
            by[gs] = (0.5 - xy[1] if flip_y else xy[1] - 0.5) * PITCH_W
        seen += 1
    keep = frame > 0
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{match}_{HALF_NAME[half]}_ball.npz"
    np.savez_compressed(dst, frame=frame[keep], x=bx[keep], y=by[keep],
                        status=status[keep], y_flipped=np.bool_(flip_y))
    return {"match": match, "half": half, "frames": int(keep.sum()),
            "missing_xy": int(np.isnan(bx[keep]).sum()),
            "raw_rows_in_range": seen, "flip_y": flip_y, "out": str(dst)}


def check_convention(data: Path, tracks: Path, table: dict, n_sample: int = 400) -> dict:
    """Decide the ball's y convention per match by putting it at somebody's feet.

    Re-derived from scratch rather than inherited from the players: in 132877 the raw ball is
    known to disagree with its own player streams, so the players' answer does not transfer.
    """
    print("BALL y CONVENTION -- median distance from the ball to the NEAREST player.")
    print("The correct choice must put the ball at a player's feet during open play.\n")
    print(f'{"match":8} {"half":5} {"n":>5} {"as-is":>8} {"flipped":>8}  choice')
    decision: dict[str, bool] = {}
    for m in MATCHES:
        xml = (data / "raw" / m / f"{m}_tracker_box_data.xml").read_bytes()
        votes = []
        for half in (1, 2):
            spec = table["matches"][m]["periods"][str(half)]
            off, n_frames = spec["gsr_to_raw_offset"], spec["n_frames"]
            d = np.load(tracks / f"{m}_{HALF_NAME[half]}_tracks.npz")
            pl = np.unique(d["player_id"])
            col = {int(p): i for i, p in enumerate(pl.tolist())}
            X = np.full((n_frames + 2, pl.size), np.nan, np.float32)
            Y = np.full_like(X, np.nan)
            idx = [col[int(p)] for p in d["player_id"].tolist()]
            ok = (d["frame"] >= 1) & (d["frame"] <= n_frames)
            X[d["frame"][ok], np.array(idx)[ok]] = d["x"][ok]
            Y[d["frame"][ok], np.array(idx)[ok]] = d["y"][ok]
            a, f, seen = [], [], 0
            for fn, _mt, st, xy, _body in _raw_rows(xml, half):
                gs = fn - off
                if xy is None or not (1 <= gs <= n_frames):
                    continue
                seen += 1
                if seen % 97 or len(a) >= n_sample:
                    continue
                px, py = (xy[0] - 0.5) * PITCH_L, (xy[1] - 0.5) * PITCH_W
                sel = ~np.isnan(X[gs])
                if sel.sum() < 15:
                    continue
                qx, qy = X[gs][sel], Y[gs][sel]
                a.append(np.min(np.hypot(qx - px, qy - py)))
                f.append(np.min(np.hypot(qx - px, qy + py)))
            if not a:
                print(f'{m:8} {HALF_NAME[half]:5} no samples'); continue
            ma, mf = float(np.median(a)), float(np.median(f))
            votes.append(mf < ma)
            print(f'{m:8} {HALF_NAME[half]:5} {len(a):5} {ma:8.2f} {mf:8.2f}  '
                  f'{"FLIPPED" if mf < ma else "as-is"}')
        decision[m] = sum(votes) > len(votes) / 2
        del xml
    print()
    print("per-match ball y_flip:", {k: v for k, v in decision.items()})
    return decision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--tracks", default="/data/share/SoccerTrack-v2/data/derived/bas/tracks")
    ap.add_argument("--out", default="/data/share/SoccerTrack-v2/data/derived/bas/ball")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="only decide and print the y convention, write nothing")
    a = ap.parse_args()
    data, tracks, out = Path(a.data), Path(a.tracks), Path(a.out)
    table = json.loads(Path(a.periods).read_text())

    stale = [m for m in MATCHES for h in (1, 2)
             if table["matches"][m]["periods"][str(h)].get("gsr_to_raw_offset") is None]
    if stale:
        print("configs/bas_periods.json has no measured gsr_to_raw_offset. Run:\n"
              "  python scripts/bas/measure_t0.py --all --write", file=sys.stderr)
        return 2

    decision = check_convention(data, tracks, table)
    if a.check:
        return 0
    if not a.all:
        ap.error("give --all or --check")

    print()
    for m in MATCHES:
        xml = (data / "raw" / m / f"{m}_tracker_box_data.xml").read_bytes()
        for half in (1, 2):
            r = extract_half(xml, m, half, table["matches"][m]["periods"][str(half)],
                             out, decision[m])
            print(f'  {m} {HALF_NAME[half]}: {r["frames"]:,} frames, '
                  f'{r["missing_xy"]} missing ball, y_flip={r["flip_y"]}', flush=True)
        del xml
    return 0


if __name__ == "__main__":
    sys.exit(main())

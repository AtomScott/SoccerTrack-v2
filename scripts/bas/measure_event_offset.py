"""Measure, per half, the time offset between the BAS event clock and the GSR frame clock.

WHY THIS EXISTS -- AND WHY THE EARLIER ANSWER WAS WRONG
    An event's frame index is `1 + (position - t0_ms)/40`. Two DIFFERENT questions hide in
    that formula and I conflated them:

      (a) which GSR frame corresponds to a given raw-tracking frameNumber, and
      (b) which GSR frame corresponds to a given BAS `position`.

    scripts/bas/measure_t0.py answers (a) exactly -- zero residual, all twenty halves --
    because the released GSR positions ARE the raw positions, so the correct alignment makes
    them identical. It says nothing about (b). BAS events are a separate annotation pass and
    need not share the raw file's `matchTime` origin.

    For the eight matches whose metadata declares `<period matchTimeStart=>`, (a) and (b)
    agreed and the distinction did not bite. For 132831 and 132877, which declare no periods,
    taking (a) as the answer to (b) moved 132831's mAP@1s from 0.338 to 0.266 -- i.e. it made
    things worse, which is the evidence that the two clocks differ there by about a second.

    Picking whichever offset scores better on the test split would be tuning on the test set.
    This script measures (b) from the data instead.

THE INSTRUMENT: A STRUCK BALL
    The touchline check in audit_annotations.py cannot resolve this. A throw-in taker stands
    on the line for seconds either side of the throw, so the statistic is flat across
    +/- 2 s -- it was built to catch a gross error and it did, but it was never sharp enough
    for a one-second one, and it passed at z = 13-21 on exactly the two misaligned matches.

    The ball is sharp. At a Pass, Shot, Cross or High Pass the ball is struck, so its speed
    STEPS within a frame or two. Averaged over the thousands of such events in a half, the
    step locates the event clock to a few frames. The statistic is

        S(d) = mean over events of [ mean ball speed over (t+d, t+d+W)
                                   - mean ball speed over (t+d-W, t+d) ]

    maximised over the candidate shift d, with W = 0.4 s. A positive step means the ball
    accelerated: that is contact.

    Only classes whose defining moment is a strike are used. Drive is excluded (the ball is
    already moving with the carrier), as are Out, Throw In, Free Kick, Goal, Header, Ball
    Player Block and Player Successful Tackle -- all either not strikes or too rare to
    average.

VALIDATION
    --validate injects a known shift into the event times and requires the estimator to
    recover it. An estimator that cannot recover a planted offset cannot be trusted to find
    a real one; this is the check the touchline test never had.

USAGE
    python scripts/bas/measure_event_offset.py --all
    python scripts/bas/measure_event_offset.py --validate
    python scripts/bas/measure_event_offset.py --all --write configs/bas_periods.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
HALF_NAME = {1: "1st", 2: "2nd"}
FRAME_MS = 40.0
FPS = 25

STRIKE_CLASSES = ("Pass", "Shot", "Cross", "High Pass")
BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
_CANON = {l.casefold(): l for l in BAS_LABELS}

WIN_FRAMES = 10        # 0.4 s either side of the candidate event time
SEARCH_FRAMES = 75     # +/- 3 s
BALL_SPEED_HALFWIN = 2  # ball moves fast; +/-2 frames is enough and stays sharp


def ball_speed(bx: np.ndarray, by: np.ndarray, k: int = BALL_SPEED_HALFWIN) -> np.ndarray:
    """Ball speed in m/s, centred difference over +/-k frames."""
    sp = np.full(bx.size, np.nan, np.float32)
    dt = 2 * k / FPS
    sp[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) / dt
    return sp


def event_rows(data: Path, match: str, half: int, spec: dict) -> np.ndarray:
    """GSR frame index of every strike-class event in this half, under the CURRENT t0."""
    path = data / "production" / "bas" / match / f"{match}_12_class_events.json"
    doc = json.loads(path.read_text())
    acts = doc.get("actions", doc.get("annotations"))
    out = []
    for a in acts:
        gt = str(a["gameTime"])
        if " - " not in gt:
            continue
        head = gt.split(" - ", 1)[0].strip()
        if not head.isdigit() or int(head) != half:
            continue
        if _CANON[str(a["label"]).casefold()] not in STRIKE_CLASSES:
            continue
        f = int(round((int(a["position"]) - spec["t0_ms"]) / FRAME_MS)) + 1
        if 1 <= f <= spec["n_frames"]:
            out.append(f)
    return np.array(sorted(out), np.int64)


def measure_half(ball_dir: Path, data: Path, match: str, half: int, spec: dict,
                 inject: int = 0) -> dict:
    b = np.load(ball_dir / f"{match}_{HALF_NAME[half]}_ball.npz")
    frame, bx, by = b["frame"], b["x"], b["y"]
    n = int(frame.max())
    X = np.full(n + 2, np.nan, np.float32); Y = np.full(n + 2, np.nan, np.float32)
    X[frame] = bx; Y[frame] = by
    sp = ball_speed(X, Y)

    rows = event_rows(data, match, half, spec) + inject
    rows = rows[(rows > WIN_FRAMES + 4) & (rows < n - WIN_FRAMES - 4)]
    if rows.size < 50:
        return {"match": match, "half": half, "ok": False, "reason": "too few strike events"}

    shifts = np.arange(-SEARCH_FRAMES, SEARCH_FRAMES + 1)
    curve = np.full(shifts.size, np.nan)
    for i, d in enumerate(shifts):
        r = rows + d
        r = r[(r > WIN_FRAMES) & (r < n - WIN_FRAMES)]
        if r.size < 50:
            continue
        pre = np.stack([sp[r - WIN_FRAMES + j] for j in range(WIN_FRAMES)])
        post = np.stack([sp[r + j] for j in range(WIN_FRAMES)])
        with np.errstate(invalid="ignore"):
            curve[i] = np.nanmean(np.nanmean(post, 0) - np.nanmean(pre, 0))
    if np.all(np.isnan(curve)):
        return {"match": match, "half": half, "ok": False, "reason": "no valid shift"}

    best = int(np.nanargmax(curve))
    d_best = int(shifts[best])
    peak = float(curve[best])
    med = float(np.nanmedian(curve))
    sd = float(np.nanstd(curve))
    z = (peak - med) / (sd + 1e-9)
    # A shift of d frames means the events sit d frames EARLIER than the ball says, so t0
    # must move by -d frames to compensate.
    t0_corr = -d_best * FRAME_MS
    return {"match": match, "half": half, "ok": bool(abs(z) > 3.0 and peak > 0),
            "reason": "" if abs(z) > 3.0 else "peak not prominent",
            "n_events": int(rows.size), "shift_frames": d_best,
            "shift_ms": d_best * FRAME_MS, "peak_step_ms": round(peak, 3),
            "z": round(z, 2), "t0_correction_ms": t0_corr,
            "t0_current": spec["t0_ms"], "t0_implied": spec["t0_ms"] + t0_corr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--ball", default="/data/share/SoccerTrack-v2/data/derived/bas/ball")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--write", nargs="?", const="configs/bas_periods.json", default=None)
    a = ap.parse_args()
    data, ball = Path(a.data), Path(a.ball)
    table = json.loads(Path(a.periods).read_text())

    if a.validate:
        print("VALIDATION -- plant a known shift in the event times and recover it.")
        print("An estimator that cannot find a planted offset cannot be trusted with a real")
        print("one. Run on three halves whose ball data is dense.\n")
        # LINEARITY is the property to test, not a zero reading. The intrinsic offset is
        # NOT zero -- the annotated event time leads the ball's speed step by about a second
        # on every match -- so requiring recovered == -planted would fail a working
        # estimator, which is exactly what it did on the first attempt.
        ok_all = True
        for m, h in (("128057", 1), ("117093", 1), ("118576", 2)):
            spec = table["matches"][m]["periods"][str(h)]
            base = measure_half(ball, data, m, h, spec, inject=0).get("shift_frames")
            for planted in (-25, +25, +50):
                r = measure_half(ball, data, m, h, spec, inject=planted)
                got = r.get("shift_frames")
                good = got is not None and base is not None and abs(got - (base - planted)) <= 2
                ok_all &= good
                print(f'  {m} {HALF_NAME[h]}  intrinsic {base:+4d}, planted {planted:+4d} '
                      f'-> recovered {got:+4d}, expected {base - planted:+4d} '
                      f'(z={r.get("z")})  {"OK" if good else "MISMATCH"}')
        print("\nestimator recovers planted offsets" if ok_all
              else "\nESTIMATOR NOT TRUSTWORTHY -- do not use")
        return 0 if ok_all else 1

    if not a.all:
        ap.error("give --all or --validate")

    print("BAS event clock vs GSR frame clock, measured from the ball's speed step at a strike.")
    print("shift = where the true event sits relative to the current mapping.\n")
    print(f'{"match":8} {"half":5} {"n_ev":>5} {"shift":>6} {"ms":>7} {"step m/s":>9} '
          f'{"z":>6}  {"t0 now":>9} {"t0 implied":>11}  verdict')
    rows = []
    for m in MATCHES:
        for half in (1, 2):
            spec = table["matches"][m]["periods"][str(half)]
            r = measure_half(ball, data, m, half, spec)
            rows.append(r)
            if "shift_frames" not in r:
                print(f'{m:8} {HALF_NAME[half]:5} FAILED: {r["reason"]}'); continue
            print(f'{m:8} {HALF_NAME[half]:5} {r["n_events"]:5} {r["shift_frames"]:6} '
                  f'{r["shift_ms"]:7.0f} {r["peak_step_ms"]:9.3f} {r["z"]:6.2f}  '
                  f'{r["t0_current"]:9.0f} {r["t0_implied"]:11.0f}  '
                  f'{"OK" if r["ok"] else "WEAK"}', flush=True)
    print()
    big = [r for r in rows if r.get("ok") and abs(r["shift_ms"]) >= 200]
    if big:
        print("HALVES WHOSE EVENT CLOCK DISAGREES WITH THE CURRENT MAPPING BY >= 200 ms:")
        for r in big:
            print(f'  {r["match"]} {HALF_NAME[r["half"]]}: {r["shift_ms"]:+.0f} ms '
                  f'-> t0 {r["t0_current"]:.0f} should be {r["t0_implied"]:.0f}')
    else:
        print("every half agrees with the current mapping to within 200 ms")

    if a.write:
        p = Path(a.write)
        for r in rows:
            if not r.get("ok"):
                continue
            spec = table["matches"][r["match"]]["periods"][str(r["half"])]
            spec["t0_ms"] = float(r["t0_implied"])
            spec["t0_source"] = "measured_vs_ball_strike"
            spec["t0_event_shift_ms"] = r["shift_ms"]
            spec["t0_event_z"] = r["z"]
        p.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

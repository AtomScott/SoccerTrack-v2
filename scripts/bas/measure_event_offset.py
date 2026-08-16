"""Measure, per half, the offset between the BAS event clock and the GSR frame numbering.

WHAT THIS FOUND
    Fourteen of the twenty halves carry a ONE-SECOND offset between the event annotations
    and the GSR frame numbering. Six do not. There is nothing in between: measured peaks are
    0, 0, -1, -1, -2, -2 on six halves and +22 to +25 on the other fourteen.

    That 6/14 split is the same one the GSR side found for the VIDEOS -- "missing frames
    from the start of the annotated period, 0 for six halves and ~25 for fourteen". The same
    per-half bookkeeping slip shows up in both places, so it is a property of the release,
    not of either task.

    It is not a semantic lag. A lag between "the annotated moment" and "the ball is struck"
    would vary continuously across halves; a discrete choice between 0 and 25 frames does not.

WHY THE EARLIER ATTEMPTS MISSED IT
    Three anchors were tried before this one, and each failed for an instructive reason:

      * the throw-in touchline check (audit_annotations.py) is flat across +/-2 s, because a
        thrower stands on the line for seconds either side. It catches a gross error, which
        is what it was built for, and passed at z = 13-21 on halves that were a second out.
      * ball SPEED steps at a strike. The estimator validates -- it recovers planted offsets
        exactly -- but the annotated moment need not be the moment of contact, so the reading
        confounds a real clock offset with an unknown semantic lag.
      * the ball crossing the pitch boundary at an Out. Peak hit rate is only 0.05-0.43, and
        exactly 0.00 for 132831 and 132877, whose ball is clamped to the pitch rectangle.

    Deriving t0 from <match>_tracker_box_data.xml (see measure_t0.py) fixes a DIFFERENT
    quantity -- which GSR frame a raw tracking frameNumber denotes -- and is exact. It does
    not answer this question, and using it as though it did made 132831 worse.

THE ANCHOR THAT WORKS: IS THE ANNOTATED ACTOR THE PLAYER NEAREST THE BALL?
    99.5% of events name their actor, and at the moment of a ball event that actor has the
    ball. The statistic is the fraction of events for which the annotated `player_id` is the
    closest player to the ball, swept over candidate shifts.

    It is sharp -- 0.86 to 0.95 at the peak against a 0.13 to 0.25 floor -- and, unlike the
    ball-speed step, it is anchored on the actor, so it does not care whether the annotation
    marks the first touch or the release: the actor has the ball either way.

    Only classes where the actor is in possession are used. Out is excluded (the last toucher
    need not be near the ball once it has gone), as are Ball Player Block and Player
    Successful Tackle (two players contest the ball, so "nearest" is ambiguous) and the rare
    classes.

VALIDATION
    --validate plants a known shift in the event times and requires the estimator to recover
    it. Linearity is what is tested, not a zero reading.

USAGE
    python scripts/bas/measure_event_offset.py --validate
    python scripts/bas/measure_event_offset.py --all
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

BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
_CANON = {l.casefold(): l for l in BAS_LABELS}
# Classes where the annotated actor is the player in possession.
POSSESSION_CLASSES = {"Pass", "Drive", "High Pass", "Cross", "Shot"}

SEARCH = 60          # +/- 2.4 s
MIN_EVENTS = 100
MIN_PEAK = 0.45      # a half whose best reading is below this is not trusted
MIN_PROMINENCE = 2.0  # peak must be this many times the floor


def _load_half(ball_dir: Path, tracks_dir: Path, match: str, half: int):
    hn = HALF_NAME[half]
    b = np.load(ball_dir / f"{match}_{hn}_ball.npz")
    d = np.load(tracks_dir / f"{match}_{hn}_tracks.npz")
    n = int(min(b["frame"].max(), d["frame"].max()))
    BX = np.full(n + 2, np.nan, np.float32)
    BY = np.full(n + 2, np.nan, np.float32)
    ok = b["frame"] <= n
    BX[b["frame"][ok]] = b["x"][ok]
    BY[b["frame"][ok]] = b["y"][ok]
    pl = np.unique(d["player_id"])
    col = {int(p): i for i, p in enumerate(pl.tolist())}
    X = np.full((n + 2, pl.size), np.nan, np.float32)
    Y = np.full_like(X, np.nan)
    sel = d["frame"] <= n
    idx = np.array([col[int(p)] for p in d["player_id"].tolist()])
    X[d["frame"][sel], idx[sel]] = d["x"][sel]
    Y[d["frame"][sel], idx[sel]] = d["y"][sel]
    return BX, BY, X, Y, col, n


def measure_half(data: Path, ball_dir: Path, tracks_dir: Path, match: str, half: int,
                 spec: dict, inject: int = 0) -> dict:
    BX, BY, X, Y, col, n = _load_half(ball_dir, tracks_dir, match, half)
    acts = json.loads(
        (data / "production" / "bas" / match / f"{match}_12_class_events.json").read_text())
    acts = acts.get("actions", acts.get("annotations"))
    ev = []
    for a in acts:
        gt = str(a["gameTime"])
        if " - " not in gt:
            continue
        head = gt.split(" - ", 1)[0].strip()
        if not head.isdigit() or int(head) != half:
            continue
        if _CANON[str(a["label"]).casefold()] not in POSSESSION_CLASSES:
            continue
        pid = a.get("player_id")
        if pid in (None, "") or int(pid) not in col:
            continue
        f = int(round((int(a["position"]) - spec["t0_ms"]) / FRAME_MS)) + 1 + inject
        if SEARCH < f < n - SEARCH:
            ev.append((f, col[int(pid)]))
    if len(ev) < MIN_EVENTS:
        return {"match": match, "half": half, "ok": False, "reason": "too few events"}

    shifts = np.arange(-SEARCH, SEARCH + 1)
    frac = np.full(shifts.size, np.nan)
    for i, s in enumerate(shifts):
        hit = tot = 0
        for f, j in ev:
            g = f + s
            dist = np.hypot(X[g] - BX[g], Y[g] - BY[g])
            if np.isnan(dist[j]) or np.all(np.isnan(dist)):
                continue
            tot += 1
            if int(np.nanargmin(dist)) == j:
                hit += 1
        if tot:
            frac[i] = hit / tot
    if np.all(np.isnan(frac)):
        return {"match": match, "half": half, "ok": False, "reason": "no valid shift"}

    pk = int(np.nanargmax(frac))
    peak_shift = int(shifts[pk])
    peak = float(frac[pk])
    floor = float(np.nanmin(frac))
    good = peak >= MIN_PEAK and peak >= MIN_PROMINENCE * max(floor, 1e-6)
    return {"match": match, "half": half, "ok": bool(good),
            "reason": "" if good else "peak not prominent enough",
            "n_events": len(ev), "shift_frames": peak_shift,
            "shift_ms": peak_shift * FRAME_MS, "peak_frac": round(peak, 4),
            "floor_frac": round(floor, 4),
            "t0_current": spec["t0_ms"],
            "t0_implied": spec["t0_ms"] - peak_shift * FRAME_MS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--ball", default="/data/share/SoccerTrack-v2/data/derived/bas/ball")
    ap.add_argument("--tracks", default="/data/share/SoccerTrack-v2/data/derived/bas/tracks")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--write", nargs="?", const="configs/bas_periods.json", default=None)
    a = ap.parse_args()
    data, ball, tracks = Path(a.data), Path(a.ball), Path(a.tracks)
    table = json.loads(Path(a.periods).read_text())

    if a.validate:
        print("VALIDATION -- plant a known shift and require the estimator to track it.")
        print("Linearity is the property under test; the intrinsic reading is not zero.\n")
        ok_all = True
        for m, h in (("128057", 1), ("117093", 2), ("118576", 1)):
            spec = table["matches"][m]["periods"][str(h)]
            base = measure_half(data, ball, tracks, m, h, spec)["shift_frames"]
            for planted in (-20, +20, +40):
                r = measure_half(data, ball, tracks, m, h, spec, inject=planted)
                got, exp = r["shift_frames"], base - planted
                good = abs(got - exp) <= 2
                ok_all &= good
                print(f'  {m} {HALF_NAME[h]}  intrinsic {base:+3d}, planted {planted:+3d} -> '
                      f'{got:+3d}, expected {exp:+3d}  {"OK" if good else "MISMATCH"}')
        print("\nestimator tracks planted offsets" if ok_all
              else "\nESTIMATOR NOT TRUSTWORTHY")
        return 0 if ok_all else 1

    if not a.all:
        ap.error("give --all or --validate")

    print("Event clock vs GSR frame numbering. shift = frames the events must move to align")
    print("with the ball; t0_implied = t0_current - shift*40.\n")
    print(f'{"match":8} {"half":5} {"n":>5} {"shift":>6} {"ms":>7} {"peak":>6} {"floor":>6}  '
          f'{"t0 now":>10} {"t0 implied":>11}  verdict')
    rows = []
    for m in MATCHES:
        for half in (1, 2):
            spec = table["matches"][m]["periods"][str(half)]
            r = measure_half(data, ball, tracks, m, half, spec)
            rows.append(r)
            if "shift_frames" not in r:
                print(f'{m:8} {HALF_NAME[half]:5} FAILED: {r["reason"]}'); continue
            print(f'{m:8} {HALF_NAME[half]:5} {r["n_events"]:5} {r["shift_frames"]:6} '
                  f'{r["shift_ms"]:7.0f} {r["peak_frac"]:6.2f} {r["floor_frac"]:6.2f}  '
                  f'{r["t0_current"]:10.0f} {r["t0_implied"]:11.0f}  '
                  f'{"OK" if r["ok"] else "WEAK"}', flush=True)
    good = [r for r in rows if r.get("ok")]
    sh = np.array([r["shift_frames"] for r in good])
    print(f'\n{len(good)}/{len(rows)} halves measured. Shift distribution:')
    near0 = [r for r in good if abs(r["shift_frames"]) <= 5]
    near25 = [r for r in good if 18 <= r["shift_frames"] <= 32]
    other = [r for r in good if r not in near0 and r not in near25]
    print(f'  ~0 frames  : {len(near0):2} halves  ' +
          ", ".join(f'{r["match"]}/{HALF_NAME[r["half"]]}' for r in near0))
    print(f'  ~25 frames : {len(near25):2} halves  ' +
          ", ".join(f'{r["match"]}/{HALF_NAME[r["half"]]}' for r in near25))
    if other:
        print(f'  other      : {len(other):2} halves  ' +
              ", ".join(f'{r["match"]}/{HALF_NAME[r["half"]]}={r["shift_frames"]}'
                        for r in other))
    print(f'  (a continuous semantic lag would not split into two clusters like this)')

    if a.write:
        p = Path(a.write)
        changed = []
        for r in rows:
            if not r.get("ok"):
                continue
            spec = table["matches"][r["match"]]["periods"][str(r["half"])]
            if abs(spec["t0_ms"] - r["t0_implied"]) > 1e-6:
                changed.append((r["match"], r["half"], spec["t0_ms"], r["t0_implied"]))
            spec["t0_ms"] = float(r["t0_implied"])
            spec["t0_source"] = "measured_vs_actor_nearest_ball"
            spec["t0_event_shift_frames"] = r["shift_frames"]
            spec["t0_peak_frac"] = r["peak_frac"]
        p.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\nwrote {p}\nt0 changed for {len(changed)} halves:")
        for m, h, o, nn in changed:
            print(f'  {m} {HALF_NAME[h]}: {o:.0f} -> {nn:.0f} ({(nn-o)/1000:+.2f} s)')
    return 0


if __name__ == "__main__":
    sys.exit(main())

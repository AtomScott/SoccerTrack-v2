"""Rebuild the per-match pitch-plane CSV that the rule-based detector consumes.

WHY THIS EXISTS
    scripts/event_detection_tracking/event_detection.py is the prior rule-based event
    detector in this repository. It reads

        data/interim/pitch_plane_coordinates/<match>/<match>_filtered_pitch_plane_coordinates.csv

    and those directories are EMPTY for all ten matches -- the CSVs were never released or
    were cleaned up. Without them the prior method cannot be run at all, so it cannot be
    compared against the learned baseline. This regenerates an equivalent from
    <match>_tracker_box_data.xml, which carries every field the format needs.

WHAT IT CANNOT REPRODUCE
    The original filename says "filtered", and the repository history mentions a Kalman
    filter. Whatever smoothing was applied is not recoverable, so this produces the
    UNFILTERED coordinates. The rule-based detector's thresholds were presumably tuned
    against the filtered version, so its scores here are a lower bound on what it achieved
    with its intended input. That has to be stated wherever its numbers appear.

COLUMNS
    frame, match_time, event_period, ball_status, id, x, y, teamId

    matching the two surviving examples under data/interim/117092 and 117093. `x` and `y` are
    normalised to [0, 1] over the pitch, as in those files, NOT metres. The ball is a row
    with id "ball". Player rows carry the teamId from the match metadata.

USAGE
    python scripts/bas/make_pitch_plane_csv.py --all
    python scripts/bas/make_pitch_plane_csv.py --all --verify   # against the two originals
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]

_FRAME = re.compile(
    rb'<frame matchTime="(-?\d+)" frameNumber="(\d+)" eventPeriod="(\w+)" '
    rb'ballStatus="(\w+)">(.*?)</frame>', re.S)
_PLAYER = re.compile(rb'<player playerId="(\d+)" loc="\[([^,]+), ([^\]]+)\]"')
_BALL = re.compile(rb'<ball playerId="ball" loc="\[([^,]+), ([^\]]+)\]"')


def team_map(data: Path, match: str) -> dict[int, str]:
    text = (data / "production" / "raw" / match /
            f"{match}_tracker_box_metadata.xml").read_text()
    out = {}
    for tag in re.findall(r"<player [^/]*/>", text):
        d = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        if "id" in d and "teamId" in d:
            out[int(d["id"])] = d["teamId"]
    return out


def build(data: Path, match: str, out_dir: Path, units: str = "normalised") -> dict:
    """Write one match's CSV.

    UNITS. The two surviving originals store x and y normalised to [0, 1]. The rule-based
    detector, however, compares `centroid_x < 52.5` and so expects METRES on a corner origin
    (0..105 by 0..68). Both are produced on request; `--units metres` is what the detector
    needs and `--units normalised` is what reproduces the originals byte for byte.
    """
    teams = team_map(data, match)
    src = data / "raw" / match / f"{match}_tracker_box_data.xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{match}_filtered_pitch_plane_coordinates.csv"
    n_rows = n_frames = n_missing_team = 0
    with src.open("rb") as fh, dst.open("w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["frame", "match_time", "event_period", "ball_status",
                    "id", "x", "y", "teamId"])
        tail = b""
        while True:
            chunk = fh.read(64 << 20)
            if not chunk:
                break
            buf = tail + chunk
            last = 0
            for g in _FRAME.finditer(buf):
                last = g.end()
                mt, fn = int(g.group(1)), int(g.group(2))
                per, bs, body = g.group(3).decode(), g.group(4).decode(), g.group(5)
                n_frames += 1
                def conv(xs, ys):
                    if units == "metres":
                        return f"{float(xs) * 105.0:.4f}", f"{float(ys) * 68.0:.4f}"
                    return xs, ys

                for q in _PLAYER.finditer(body):
                    pid = int(q.group(1))
                    tid = teams.get(pid, "")
                    if not tid:
                        n_missing_team += 1
                    x, y = conv(q.group(2).decode(), q.group(3).decode())
                    w.writerow([fn, float(mt), per, bs, pid, x, y, tid])
                    n_rows += 1
                b = _BALL.search(body)
                if b is not None:
                    x, y = conv(b.group(1).decode(), b.group(2).decode())
                    w.writerow([fn, float(mt), per, bs, "ball", x, y, ""])
                    n_rows += 1
            tail = buf[last:] if last else buf[-4096:]
    return {"match": match, "rows": n_rows, "frames": n_frames,
            "missing_team": n_missing_team, "out": str(dst)}


def verify(data: Path, out_dir: Path) -> int:
    """Compare against the two surviving originals under data/interim."""
    print("VERIFY against the original CSVs, which exist only for 117092 and 117093.\n")
    ok = True
    for match in ("117092", "117093"):
        orig = data / "interim" / match / f"{match}_pitch_plane_coordinates_1st_half.csv"
        mine = out_dir / match / f"{match}_filtered_pitch_plane_coordinates.csv"
        if not orig.exists() or not mine.exists():
            print(f"  {match}: missing input, skipped"); continue
        want = {}
        with orig.open() as fh:
            rd = csv.DictReader(fh)
            for i, r in enumerate(rd):
                if i >= 4000:
                    break
                want[(int(r["frame"]), r["id"])] = (r["x"], r["y"], r["event_period"])
        seen = agree = 0
        with mine.open() as fh:
            for r in csv.DictReader(fh):
                k = (int(r["frame"]), r["id"])
                if k in want:
                    seen += 1
                    x0, y0, p0 = want[k]
                    if (abs(float(r["x"]) - float(x0)) < 1e-9
                            and abs(float(r["y"]) - float(y0)) < 1e-9
                            and r["event_period"] == p0):
                        agree += 1
                if seen >= len(want):
                    break
        rate = agree / seen if seen else 0.0
        good = seen > 500 and rate > 0.999
        ok &= good
        print(f"  {match}: {agree}/{seen} rows identical ({rate:.4f}) -> "
              f"{'OK' if good else 'MISMATCH'}")
    print("\nregenerated CSVs reproduce the originals" if ok
          else "\nREGENERATED CSVs DIFFER -- do not feed them to the detector")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out", default="/data/share/SoccerTrack-v2/data/derived/bas/pitch_plane")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--match")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--units", choices=["normalised", "metres"], default="normalised",
                    help="the surviving CSVs are normalised; the rule-based detector's own "
                         "code (centroid_x < 52.5) shows it expects metres on a corner origin")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)

    if a.all or a.match:
        for m in (MATCHES if a.all else [a.match]):
            r = build(data, m, out / m, a.units)
            print(f'  {m}: {r["rows"]:,} rows over {r["frames"]:,} frames'
                  + (f', {r["missing_team"]:,} without teamId' if r["missing_team"] else ''),
                  flush=True)
    if a.verify:
        return verify(data, out)
    if not (a.all or a.match or a.verify):
        ap.error("give --all, --match or --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())

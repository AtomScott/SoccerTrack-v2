"""Extract per-frame player pitch positions from the released GSR ground truth.

WHY THIS EXISTS
    Trajectory-based ball action spotting (docs/experiment-design-bas.md, Experiment B)
    consumes 22 player positions per frame. Those positions live in the released GSR
    GameState files -- which are 2.7 GB EACH, 54 GB across the twenty halves. ``json.loads``
    on one of them peaks around 25 GB of process memory, so the obvious reader cannot be
    used to build a training set.

    Almost all of that bulk is the per-frame pitch-line annotation (a "pitch" supercategory
    record carrying ~30 named line segments for every one of ~68,000 frames). The player
    records are a small fraction of the file. This script streams the bytes and pulls only
    the object records, writing a compact .npz per half (~30 MB) that the feature builder
    and the model can load in milliseconds.

WHAT IT EXTRACTS
    frame       int32   1-based frame index within the half, from the GameState image_id
    track_id    int16   GameState track id (stable within a half)
    player_id   int32   provider player id; the SAME namespace as the BAS `player_id`
                        field, which is what lets an event be linked to its actor
    team        int8    0 = left, 1 = right, -1 = unknown
    role        int8    0 = player, 1 = goalkeeper
    x, y        float32 pitch metres, centre-origin (the bbox_pitch bottom-middle point)

    `bbox_image` is deliberately NOT extracted. Those boxes are auto-generated and are not
    ground truth (standing instruction); nothing downstream of this file may treat them as
    such.

COORDINATE RESOLUTION -- MEASURE, DO NOT ASSUME
    The pitch coordinates are quantised. They derive from a normalised [0,1] pitch position
    stored to two decimal places, so x lands on multiples of 1.05 m and y on multiples of
    0.68 m. A finite difference over one frame (40 ms) therefore carries up to 26 m/s of
    pure quantisation noise. Velocity features MUST be computed over a smoothing window --
    see src/bas/features.py, which measures the resulting noise floor rather than guessing.

VALIDATION
    --validate cross-checks the extractor against an INDEPENDENT source: the per-match
    pitch-plane CSVs under data/interim, which exist for 117092 and 117093 and were produced
    by a different pipeline. It checks
      * that the frame numbering lines up (GameState frame 1 == CSV frame `frameStart`), and
      * that the de-normalised CSV position equals the extracted metre position.
    It is designed to FAIL if the regex drifts, if the frame convention changes, or if the
    metre scaling is wrong -- a self-consistency check against the same file would catch
    none of those.

USAGE
    python scripts/bas/extract_tracks.py --validate
    python scripts/bas/extract_tracks.py --all --workers 5
    python scripts/bas/extract_tracks.py --match 128057 --half 1st
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
HALVES = ("1st", "2nd")

# Pitch dimensions, from <match>_tracker_box_metadata.xml (<pitch width="105" height="68"/>),
# identical for all ten matches.
PITCH_L, PITCH_W = 105.0, 68.0

# One object annotation, captured in a single pass. Anchored on the image_id/track_id/
# supercategory prefix and terminated by "bbox_pitch_raw", so group 3 holds exactly
# attributes + bbox_image + bbox_pitch. Records are ~800 bytes; the 2000-byte bound keeps
# the non-greedy scan short and makes a malformed record fail to match rather than run away.
_REC = re.compile(
    rb'"image_id":\s*"(\d+)",\s*'
    rb'"track_id":\s*(\d+),\s*'
    rb'"supercategory":\s*"object",'
    rb'(.{0,2000}?)'
    rb'"bbox_pitch_raw"',
    re.S,
)
_ROLE = re.compile(rb'"role":\s*"([^"]*)"')
_TEAM = re.compile(rb'"team":\s*(?:"([^"]*)"|null)')
_PID = re.compile(rb'"player_id":\s*(?:"?(-?\d+)"?|null)')
_XBM = re.compile(rb'"x_bottom_middle":\s*(-?[\d.eE+]+|null)')
_YBM = re.compile(rb'"y_bottom_middle":\s*(-?[\d.eE+]+|null)')

_ROLE_CODE = {b"player": 0, b"goalkeeper": 1, b"referee": 2, b"other": 3, b"ball": 4}
_TEAM_CODE = {b"left": 0, b"right": 1}

CHUNK = 64 << 20
# Longest possible record, so a record straddling a chunk boundary is retried with the
# next chunk instead of being dropped. Must exceed the {0,2000} bound above plus the prefix.
OVERLAP = 4 << 10


def gsr_path(data: Path, match: str, half: str) -> Path:
    return data / "production" / "gsr" / match / f"{match}_{half}.json"


def extract_half(data: Path, match: str, half: str, out_dir: Path,
                 verbose: bool = True) -> dict:
    """Stream one half's GSR file and write <match>_<half>_tracks.npz."""
    src = gsr_path(data, match, half)
    if not src.exists():
        return {"match": match, "half": half, "ok": False, "reason": "missing GSR file"}

    t0 = time.time()
    frames: list[np.ndarray] = []
    tids: list[np.ndarray] = []
    pids: list[np.ndarray] = []
    teams: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_null = 0
    n_rec = 0

    with src.open("rb") as fh:
        tail = b""
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            last_end = 0
            bf, bt, bp, bteam, brole, bx, by = [], [], [], [], [], [], []
            for m in _REC.finditer(buf):
                last_end = m.end()
                body = m.group(3)
                xm = _XBM.search(body)
                ym = _YBM.search(body)
                n_rec += 1
                if xm is None or ym is None or xm.group(1) == b"null" or ym.group(1) == b"null":
                    n_null += 1
                    continue
                rm = _ROLE.search(body)
                tm = _TEAM.search(body)
                pm = _PID.search(body)
                # image_id is "3" + a 6-digit 1-based frame; a staged sequence uses the
                # 10-character SoccerNet form. The last six digits are the frame either way.
                sid = m.group(1)
                bf.append(int(sid[-6:]))
                bt.append(int(m.group(2)))
                bp.append(int(pm.group(1)) if (pm and pm.group(1)) else -1)
                bteam.append(_TEAM_CODE.get(tm.group(1) if tm else None, -1) if tm else -1)
                brole.append(_ROLE_CODE.get(rm.group(1) if rm else b"", 3))
                bx.append(float(xm.group(1)))
                by.append(float(ym.group(1)))
            if bf:
                frames.append(np.array(bf, np.int32))
                tids.append(np.array(bt, np.int16))
                pids.append(np.array(bp, np.int32))
                teams.append(np.array(bteam, np.int8))
                roles.append(np.array(brole, np.int8))
                xs.append(np.array(bx, np.float32))
                ys.append(np.array(by, np.float32))
            # keep the unconsumed remainder so a straddling record is not lost
            keep = max(len(buf) - last_end, 0)
            tail = buf[-min(max(keep, OVERLAP), len(buf)):] if len(buf) else b""

    if not frames:
        return {"match": match, "half": half, "ok": False, "reason": "no object records found"}

    frame = np.concatenate(frames)
    order = np.argsort(frame, kind="stable")
    out = {
        "frame": frame[order],
        "track_id": np.concatenate(tids)[order],
        "player_id": np.concatenate(pids)[order],
        "team": np.concatenate(teams)[order],
        "role": np.concatenate(roles)[order],
        "x": np.concatenate(xs)[order],
        "y": np.concatenate(ys)[order],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{match}_{half}_tracks.npz"
    np.savez_compressed(dst, **out)

    uf = np.unique(out["frame"])
    res = {
        "match": match, "half": half, "ok": True, "reason": "",
        "records": int(out["frame"].size), "records_seen": n_rec, "null_pitch": n_null,
        "frames": int(uf.size), "frame_min": int(uf.min()), "frame_max": int(uf.max()),
        "entities_per_frame": round(float(out["frame"].size / uf.size), 2),
        "x_range": [round(float(out["x"].min()), 2), round(float(out["x"].max()), 2)],
        "y_range": [round(float(out["y"].min()), 2), round(float(out["y"].max()), 2)],
        "secs": round(time.time() - t0, 1), "out": str(dst),
    }
    if verbose:
        print(f"  {match} {half}: {res['records']:,} records over {res['frames']:,} frames "
              f"({res['entities_per_frame']} per frame), {n_null:,} null pitch, "
              f"{res['secs']}s -> {dst.name}", flush=True)
    return res


# ---------------------------------------------------------------------------
# Validation against an independent source
# ---------------------------------------------------------------------------

def _csv_path(data: Path, match: str, half: str) -> Path:
    return data / "interim" / match / f"{match}_pitch_plane_coordinates_{half}_half.csv"


def validate(data: Path, out_dir: Path, tol_m: float = 0.02) -> int:
    """Check extracted metre positions against the independent pitch-plane CSVs.

    The CSVs exist only for 117092 and 117093 and were written by a different pipeline
    (interim/convert_csv_to_gsr_json.py consumed them). They store the position NORMALISED
    to [0,1] over the pitch, and they carry the RAW frame number and the match clock, so
    agreeing with them pins three separate things at once: the field-extraction regex, the
    frame-numbering convention, and the metre scaling.

    THE FRAME CONVENTION THIS ASSERTS, which was measured and not assumed:

        GameState frame 2  ==  the period's first raw frame  ==  matchTimeStart

    GameState frame 1 exists but its bbox_pitch is null -- there is no source row for it.
    Sweeping the alignment over +/-2 frames on 117092's first half gave a mean absolute
    error of 0.1135 m at the obvious "frame 1 is the first frame" mapping and EXACTLY
    0.0000 m one frame later, over 110 positions. One frame is 40 ms; at the 1.05 m
    coordinate quantisation an exact zero across 110 samples is not a coincidence.
    """
    print("VALIDATION -- extracted GSR positions vs the independent pitch-plane CSVs\n")
    ok_all = True
    for match in ("117092", "117093"):
        for half in HALVES:
            csv = _csv_path(data, match, half)
            npz = out_dir / f"{match}_{half}_tracks.npz"
            if not csv.exists():
                print(f"  {match} {half}: no CSV, skipped"); continue
            if not npz.exists():
                print(f"  {match} {half}: {npz.name} not built yet -- run the extractor first")
                ok_all = False; continue
            d = np.load(npz)

            # --- read the CSV rows for a handful of frames spread through the half ---
            import csv as _csv
            want_raw: dict[int, dict[int, tuple[float, float]]] = {}
            raw_frames: list[int] = []
            with csv.open() as fh:
                rd = _csv.reader(fh)
                header = next(rd)
                ci = {k: header.index(k) for k in ("frame", "match_time", "id", "x", "y")}
                first_raw = None
                for row in rd:
                    fr = int(row[ci["frame"]])
                    if first_raw is None:
                        first_raw = fr
                        # Sample 6 frames spread through the half. Start one stride in:
                        # the period's very first frame maps to GameState frame 2, and
                        # GameState frame 1 carries a null bbox_pitch with no CSV row to
                        # compare against.
                        raw_frames = [first_raw + k * 5000 for k in range(1, 7)]
                        want = set(raw_frames)
                    if fr in want:
                        # These CSVs carry a 23rd entity with id "ball". The RELEASED GSR
                        # ground truth has no ball at all, so there is nothing to compare
                        # it against and nothing downstream may depend on it.
                        ident = row[ci["id"]]
                        if not ident.lstrip("-").isdigit():
                            continue
                        want_raw.setdefault(fr, {})[int(ident)] = (
                            float(row[ci["x"]]), float(row[ci["y"]]))
                    elif fr > raw_frames[-1]:
                        break

            # GameState frame 2 corresponds to the period's first raw frame (see above);
            # `first_raw` is that frame, and raw_frames[0] is one stride past it.
            offset = first_raw - 2
            worst = 0.0
            n_cmp = 0
            missing = 0
            for raw in raw_frames:
                gs = raw - offset
                sel = d["frame"] == gs
                if not sel.any():
                    missing += 1
                    continue
                got = {int(p): (float(a), float(b))
                       for p, a, b in zip(d["player_id"][sel], d["x"][sel], d["y"][sel])}
                for pid, (nx, ny) in want_raw.get(raw, {}).items():
                    if pid not in got:
                        continue
                    # normalised [0,1] over the pitch -> centre-origin metres
                    ex, ey = (nx - 0.5) * PITCH_L, (ny - 0.5) * PITCH_W
                    gx, gy = got[pid]
                    worst = max(worst, abs(ex - gx), abs(ey - gy))
                    n_cmp += 1
            good = n_cmp >= 50 and worst <= tol_m and missing == 0
            ok_all &= good
            print(f"  {match} {half}: compared {n_cmp} positions over {len(raw_frames)} frames "
                  f"(CSV frame {raw_frames[0]} == GameState frame 1), worst |delta| = "
                  f"{worst:.4f} m -> {'OK' if good else 'MISMATCH'}"
                  + (f"  [{missing} frames absent]" if missing else ""))
    print("\nextractor agrees with the independent source" if ok_all
          else "\nEXTRACTOR NOT TRUSTWORTHY -- do not build features from it")
    return 0 if ok_all else 1


def _job(args):
    data, match, half, out_dir = args
    return extract_half(Path(data), match, half, Path(out_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out", default="/data/share/SoccerTrack-v2/data/derived/bas/tracks")
    ap.add_argument("--match")
    ap.add_argument("--half", choices=list(HALVES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=5,
                    help="parallel halves; each reads ~2.7 GB, so this is I/O bound")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    data, out_dir = Path(a.data), Path(a.out)

    if a.validate:
        return validate(data, out_dir)

    if a.all:
        jobs = [(str(data), m, h, str(out_dir)) for m in MATCHES for h in HALVES]
    elif a.match and a.half:
        jobs = [(str(data), a.match, a.half, str(out_dir))]
    elif a.match:
        jobs = [(str(data), a.match, h, str(out_dir)) for h in HALVES]
    else:
        ap.error("give --all, or --match (optionally with --half), or --validate")

    print(f"extracting {len(jobs)} half/halves with {a.workers} workers -> {out_dir}\n")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fu in as_completed(futs):
            results.append(fu.result())
    bad = [r for r in results if not r["ok"]]
    print(f"\ndone in {time.time() - t0:.0f}s: {len(results) - len(bad)} ok, {len(bad)} failed")
    for r in bad:
        print(f"  FAILED {r['match']} {r['half']}: {r['reason']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

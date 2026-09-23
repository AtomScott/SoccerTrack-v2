"""Extract GSR annotation statistics from the 20 GameState GT files.

Per half: per-track (track_id) annotated-frame counts and metadata, plus a
2D histogram of bbox_pitch bottom-middle positions on the metric pitch.

Streaming regex over the raw bytes (files are uniformly pretty-printed,
generated JSON); validate_extract.py cross-checks one half against a full
json.load parse before the numbers are used anywhere.
"""
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DATA = os.environ.get("SOCCERTRACK_GSR", "/data/share/SoccerTrack-v2/release_v1_1/gsr")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsr_stats")

# Histogram: pitch is 105 x 68 m centred on (0,0). GT positions are
# quantised to a 100 x 100 grid of the pitch (steps of 1.05 m in x and
# 0.68 m in y, verified to machine precision across matches), so bins are
# aligned to that lattice with lattice points at bin centres; 3 cells of
# margin each side for out-of-play positions (throw-in takers etc.).
XR = (-56.175, 56.175)  # -52.5 - 3.5*1.05 .. +
YR = (-36.38, 36.38)    # -34.0 - 3.5*0.68 .. +
BINS = (107, 107)

REC = re.compile(
    rb'"image_id": "(\d+)",\s+'
    rb'"track_id": (\d+),\s+'
    rb'"supercategory": "object",\s+'
    rb'"category_id": \d+,\s+'
    rb'"attributes": \{\s+'
    rb'"role": "([^"]*)",\s+'
    rb'"jersey": (null|"[^"]*"),\s+'
    rb'"team": (null|"[^"]*"),\s+'
    rb'"player_id": (null|\d+)'
    rb'.*?"bbox_pitch": (null|\{.*?\})',
    re.DOTALL,
)
MID = re.compile(
    rb'"x_bottom_middle": (-?[\d.eE+-]+|null),\s+"y_bottom_middle": (-?[\d.eE+-]+|null)'
)


def process_half(path):
    with open(path, "rb") as f:
        blob = f.read()
    tracks = {}
    xs, ys = [], []
    n_obj = 0
    n_null_pitch = 0
    for m in REC.finditer(blob):
        n_obj += 1
        frame = int(m.group(1)[-6:])
        tid = int(m.group(2))
        role = m.group(3).decode()
        pid = None if m.group(6) == b"null" else int(m.group(6))
        t = tracks.get(tid)
        if t is None:
            tracks[tid] = t = {"role": role, "player_id": pid, "n": 0,
                               "fmin": frame, "fmax": frame}
        t["n"] += 1
        if frame < t["fmin"]:
            t["fmin"] = frame
        if frame > t["fmax"]:
            t["fmax"] = frame
        pitch = m.group(7)
        if pitch == b"null":
            n_null_pitch += 1
            continue
        mm = MID.search(pitch)
        if mm is None or mm.group(1) == b"null" or mm.group(2) == b"null":
            n_null_pitch += 1
            continue
        xs.append(float(mm.group(1)))
        ys.append(float(mm.group(2)))
    del blob
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    hist, _, _ = np.histogram2d(xs, ys, bins=BINS, range=[XR, YR])
    out_of_range = int(xs.size - hist.sum())
    half = os.path.basename(path).replace(".json", "")
    np.save(os.path.join(OUT, f"{half}_hist.npy"), hist.astype(np.int64))
    summary = {
        "half": half,
        "n_object_annotations": n_obj,
        "n_null_pitch": n_null_pitch,
        "n_positions": int(xs.size),
        "n_out_of_hist_range": out_of_range,
        "x_min": float(xs.min()), "x_max": float(xs.max()),
        "y_min": float(ys.min()), "y_max": float(ys.max()),
        "tracks": {str(k): v for k, v in sorted(tracks.items())},
    }
    with open(os.path.join(OUT, f"{half}_tracks.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return half, n_obj, len(tracks), n_null_pitch, out_of_range


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(
        os.path.join(DATA, d, fn)
        for d in os.listdir(DATA)
        for fn in os.listdir(os.path.join(DATA, d))
        if re.fullmatch(r"\d+_(1st|2nd)\.json", fn)
    )
    assert len(paths) == 20, paths
    with ProcessPoolExecutor(max_workers=6) as ex:
        for half, n_obj, n_tracks, n_null, n_oor in ex.map(process_half, paths):
            print(f"{half}: {n_obj:,} object anns, {n_tracks} tracks, "
                  f"{n_null} null-pitch, {n_oor} outside hist range",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())

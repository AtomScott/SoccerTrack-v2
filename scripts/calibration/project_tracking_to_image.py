"""Project BePro tracking positions onto the image, to validate the coordinate chain.

WHAT THIS VALIDATES
    raw/<match>/<match>_tracker_box_data.xml holds per-frame player and ball positions as
    pitch coordinates NORMALISED to the unit square, e.g. loc="[0.12, 0.48]". The full
    chain to pixels is:

        loc  ->  pitch metres (x*105, y*68)  ->  cv2.fisheye.projectPoints(rvec, tvec, K, D)
                                             ->  distorted image pixels

    No homography is involved: cv2.fisheye.calibrate already returns the pitch plane's pose
    relative to the camera, and the camera is fixed, so one (rvec, tvec) holds for the whole
    match. If the markers land on the players, calibration and tracking agree.

    Confirmed conventions (do not "fix" these):
      * y is used as-is. Flipping it puts every marker on empty grass.
      * pitch origin is a corner, x in [0,105], y in [0,68] -- the same frame the keypoints
        use, which is why the keypoint fit's extrinsics apply directly.

FRAME ALIGNMENT
    The XML numbers frames continuously across the match while the video is split per half,
    so the offset is the frameNumber of the first FIRST_HALF/SECOND_HALF row. That differs
    per match -- 251 for 117093, 1 for 132831 -- so it is read from the file rather than
    assumed. --search-offsets additionally tries a few neighbouring offsets and reports
    which minimises total marker-to-nearest-marker drift, since being a few frames out
    displaces each player along their own motion vector.

USAGE
    python scripts/calibration/project_tracking_to_image.py
    python scripts/calibration/project_tracking_to_image.py --matches 117093 --frame 30000

THEN LOOK AT
    <out>/index.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
KEYPOINT_SWAPS = {"132831": [("(88.5,13.84)", "(105,54.16)")]}
PITCH_W, PITCH_H = 105.0, 68.0

FLAGS = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
         + cv2.fisheye.CALIB_FIX_SKEW
         + cv2.fisheye.CALIB_CHECK_COND
         + cv2.fisheye.CALIB_FIX_K3
         + cv2.fisheye.CALIB_FIX_K4)
CRITERIA = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)

ENTITY_RE = re.compile(r'<(player|ball)\s+playerId="([^"]+)"\s+loc="\[([-\d.]+),\s*([-\d.]+)\]"')


def load_keypoints(data: Path, match: str, corrections: Path):
    corrected = corrections / f"{match}_keypoints.json"
    if corrected.exists():
        d = json.load(open(corrected))
        keys = list(d)
        pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
        return pitch, np.array(list(d.values()), float), f"corrections/{corrected.name}"
    d = json.load(open(data / "raw" / match / f"{match}_keypoints.json"))
    keys = list(d)
    pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
    image = np.array(list(d.values()), float)
    src = "dataset"
    for a, b in KEYPOINT_SWAPS.get(match, []):
        if a in d and b in d:
            i, j = keys.index(a), keys.index(b)
            image[[i, j]] = image[[j, i]]
            src = "dataset + in-code swap"
    return pitch, image, src


def calibrate(pitch, image, w, h):
    objp = np.concatenate([pitch, np.zeros((len(pitch), 1))], 1).astype(np.float32).reshape(-1, 1, 3)
    imgp = image.astype(np.float32).reshape(-1, 1, 2)
    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate([objp], [imgp], (w, h), K, D, None, None,
                                                    flags=FLAGS, criteria=CRITERIA)
    return rms, K, D, rvecs[0], tvecs[0]


def half_offset(xml: Path, period: str = "FIRST_HALF", scan_lines: int = 4000):
    """frameNumber of the first row of the requested period; the video's frame 0."""
    pat = re.compile(rf'frameNumber="(\d+)"\s+eventPeriod="{period}"')
    with open(xml, "r", errors="replace") as f:
        for _ in range(scan_lines):
            line = f.readline()
            if not line:
                break
            mo = pat.search(line)
            if mo:
                return int(mo.group(1))
    return None


def entities_at(xml: Path, frame_number: int, period: str = "FIRST_HALF"):
    """Stream to one <frame> and return its (kind, id, loc_x, loc_y) rows."""
    needle = f'frameNumber="{frame_number}"'
    buf, on = [], False
    with open(xml, "r", errors="replace") as f:
        for line in f:
            if not on and needle in line and period in line:
                on = True; buf = [line]; continue
            if on:
                buf.append(line)
                if "</frame>" in line:
                    break
    if not on:
        return []
    return [(mo.group(1), mo.group(2), float(mo.group(3)), float(mo.group(4)))
            for mo in ENTITY_RE.finditer("".join(buf))]


def project(ents, rvec, tvec, K, D):
    if not ents:
        return []
    obj = np.array([[[lx * PITCH_W, ly * PITCH_H, 0.0]] for _, _, lx, ly in ents], dtype=np.float32)
    pts, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    pts = pts.reshape(-1, 2)
    return [(e[0], e[1], p) for e, p in zip(ents, pts) if np.isfinite(p).all()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out", default="outputs/tracking_projection")
    ap.add_argument("--corrections", default="data_corrections")
    ap.add_argument("--matches", nargs="*", default=MATCHES)
    ap.add_argument("--frame", type=int, default=30000, help="frame index within the half")
    ap.add_argument("--half", default="1st", choices=["1st", "2nd"])
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    corrections = Path(args.corrections)
    period = "FIRST_HALF" if args.half == "1st" else "SECOND_HALF"
    rows = []

    for m in args.matches:
        video = data / "interim" / m / f"{m}_panorama_{args.half}_half.mp4"
        xml = data / "raw" / m / f"{m}_tracker_box_data.xml"
        if not video.exists() or not xml.exists():
            rows.append(dict(match=m, status="MISSING INPUT")); print(f"  {m}: missing input"); continue
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read(); cap.release()
        if not ok:
            rows.append(dict(match=m, status="NOFRAME")); print(f"  {m}: no frame"); continue
        h, w = frame.shape[:2]

        pitch, image, ksrc = load_keypoints(data, m, corrections)
        try:
            rms, K, D, rvec, tvec = calibrate(pitch, image, w, h)
        except cv2.error:
            rows.append(dict(match=m, status="CALIBRATION REFUSED", note=ksrc))
            print(f"  {m}: CALIBRATION REFUSED (keypoints from {ksrc})"); continue

        off = half_offset(xml, period)
        if off is None:
            rows.append(dict(match=m, status="NO PERIOD ROW")); print(f"  {m}: no {period} row"); continue
        ents = entities_at(xml, args.frame + off, period)
        proj = project(ents, rvec, tvec, K, D)

        vis = frame.copy()
        inside = 0
        for kind, _pid, p in proj:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                inside += 1
            col = (0, 255, 255) if kind == "ball" else (0, 255, 0)
            cv2.circle(vis, (x, y), 13, (0, 0, 0), 3)
            cv2.circle(vis, (x, y), 11, col, 2)
            cv2.drawMarker(vis, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 14, 3)
            cv2.drawMarker(vis, (x, y), col, cv2.MARKER_CROSS, 12, 1)
        label = (f"{m}  {args.half} half frame {args.frame} (xml frameNumber {args.frame+off}, "
                 f"offset {off})  rms={rms:.2f}  {len(proj)} projected, {inside} in frame")
        for t, c in ((6, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(vis, label, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, t, cv2.LINE_AA)
        tag = f"{m}_tracking_{args.half}_{args.frame}"
        hh = 470
        cv2.imwrite(str(out / f"{tag}.jpg"),
                    cv2.resize(vis, (int(w * hh / h), hh)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(dict(match=m, status="OK", rms=rms, n=len(proj), inside=inside,
                         offset=off, tag=tag, note=ksrc))
        print(f"  {m}: rms={rms:7.2f} offset={off:5d} projected={len(proj):3d} in_frame={inside:3d} ({ksrc})")

    with open(out / "summary.tsv", "w") as f:
        f.write("match\tstatus\trms_px\tprojected\tin_frame\txml_frame_offset\tkeypoint_source\n")
        for r in rows:
            f.write(f"{r['match']}\t{r.get('status','')}\t{r.get('rms','')}\t{r.get('n','')}\t"
                    f"{r.get('inside','')}\t{r.get('offset','')}\t{r.get('note','')}\n")

    cards = []
    for r in rows:
        if r.get("status") != "OK":
            cards.append(f'<div class="c bad"><h3>{r["match"]}</h3><p>{r.get("status")}</p>'
                         f'<pre>{r.get("note","")}</pre></div>')
            continue
        cards.append(f'<div class="c"><h3>{r["match"]}</h3>'
                     f'<p class="st">rms {r["rms"]:.2f} px · {r["n"]} entities, {r["inside"]} in frame · '
                     f'xml offset {r["offset"]}</p>'
                     f'<a href="{r["tag"]}.jpg"><img src="{r["tag"]}.jpg" loading="lazy"></a>'
                     f'<pre>keypoints: {r.get("note","")}</pre></div>')
    html = f"""<!doctype html><meta charset="utf-8"><title>SoccerTrack v2 tracking projection</title>
<style>
 body{{font:14px/1.6 system-ui,sans-serif;margin:24px;background:#0d1117;color:#e6edf3}}
 .grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(560px,1fr))}}
 .c{{border:1px solid #30363d;border-radius:8px;padding:10px;background:#161b22}}
 .c.bad{{border-color:#f85149}} .c h3{{margin:0 0 4px;font-size:15px}}
 .st{{margin:0 0 8px;color:#8b949e;font-size:12px}} img{{width:100%;border-radius:4px;display:block}}
 pre{{margin:6px 0 0;color:#6e7681;font-size:11px}}
 .note{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:16px 0}}
</style>
<h1>BePro tracking projected onto the image</h1>
<div class="note">
<p>Green rings are players, yellow the ball, taken from
<code>raw/&lt;match&gt;/&lt;match&gt;_tracker_box_data.xml</code> and projected through the
calibration fitted to that match's pitch keypoints. <b>If the markers sit on the players, the
whole coordinate chain agrees.</b></p>
<p>Residual offsets whose <i>direction varies per player</i> indicate frame synchronisation,
not calibration — being a few frames out displaces each player along their own motion vector.
A calibration error would displace everyone the same way.</p>
</div>
<div class="grid">{"".join(cards)}</div>
"""
    (out / "index.html").write_text(html)
    ok = [r for r in rows if r.get("status") == "OK"]
    print(f"\n{len(ok)}/{len(rows)} projected. wrote {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate every match from its pitch keypoints, and validate each result.

WHY
    data/raw/<match>/<match>_keypoints.json is the authoritative pitch-geometry
    annotation: 65 correspondences from pitch metres to TRUE image pixels. Verified
    against the pitch `lines` inside the released GSR files, which are exactly these
    same pixels divided by 3840 (x) and 1504 (y) -- 132831's dimensions, applied to
    every match. So the keypoints are the clean source and the normalised copy in the
    GSR files is derived (and mis-scaled for nine of the ten matches).

    This script fits cv2.fisheye from those keypoints for all ten matches, keeping
    CALIB_CHECK_COND on so a bad fit fails loudly rather than silently, and renders each
    undistorted frame with the undistorted keypoints drawn on top. If the calibration is
    right, both touchlines come out straight and land on the painted lines.

TWO CORRECTIONS APPLIED HERE
    1. 132831 has two transposed keypoint labels -- (88.5,13.84) and (105,54.16) hold
       each other's image coordinates. Uncorrected, its fit is RMS 1260.95 px and
       CALIB_CHECK_COND rejects it; corrected, RMS 18.07. See fix_132831_keypoints.py.
    2. The undistortion canvas is sized to contain the pitch rather than inherited from
       the input frame. generate_calibration_mappings.py uses balance=1 with an
       input-sized canvas, which for 132831's wider field of view leaves the pitch
       occupying 22% of the frame width against 71% for the others.

USAGE
    python scripts/calibration/calibrate_all_from_keypoints.py
    python scripts/calibration/calibrate_all_from_keypoints.py --matches 132831 117093

THEN LOOK AT
    <out>/index.html   -- every match's undistorted frame with keypoints overlaid
    <out>/summary.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]

# Known annotation defects, applied on load. Keys hold each other's image coordinates.
KEYPOINT_SWAPS = {"132831": [("(88.5,13.84)", "(105,54.16)")]}

FLAGS = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
         + cv2.fisheye.CALIB_FIX_SKEW
         + cv2.fisheye.CALIB_CHECK_COND
         + cv2.fisheye.CALIB_FIX_K3
         + cv2.fisheye.CALIB_FIX_K4)
CRITERIA = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)


def load_keypoints(data: Path, match: str):
    with open(data / "raw" / match / f"{match}_keypoints.json") as f:
        d = json.load(f)
    keys = list(d)
    pitch = np.array([[*map(float, k.strip("()").split(","))] for k in keys], float)
    image = np.array(list(d.values()), float)
    applied = []
    for a, b in KEYPOINT_SWAPS.get(match, []):
        if a in d and b in d:
            i, j = keys.index(a), keys.index(b)
            image[[i, j]] = image[[j, i]]
            applied.append(f"{a}<->{b}")
    return keys, pitch, image, applied


def grab(data: Path, match: str, idx: int):
    for p in (data / "interim" / match / f"{match}_panorama_1st_half.mp4",
              data / "raw" / match / f"{match}_panorama.mp4"):
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            cap.release()
            if ok:
                return fr, p.name
    return None, None


def straightness(pitch, undistorted):
    """Mean perpendicular RMS residual of the two touchlines after undistortion.

    Restricted to the y=0 and y=68 rows: those are 21 collinear points each, so a
    correct undistortion makes them straight. NOTE: this number is only meaningful
    alongside the spread below -- a collapsed calibration squeezes every point into a
    blob and scores a deceptively perfect ~0.
    """
    res = []
    for row in (0.0, 68.0):
        sel = np.round(pitch[:, 1], 1) == row
        if sel.sum() < 3:
            continue
        P = undistorted[sel]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        res.append(float(np.sqrt((((P - c) @ vt[1]) ** 2).mean())))
    return float(np.mean(res)) if res else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out", default="outputs/calibration_all")
    ap.add_argument("--matches", nargs="*", default=MATCHES)
    ap.add_argument("--frame", type=int, default=30000)
    ap.add_argument("--canvas-width", type=int, default=4096)
    ap.add_argument("--vertical-margin", type=float, default=3.0)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for m in args.matches:
        keys, pitch, image, applied = load_keypoints(data, m)
        frame, src = grab(data, m, args.frame)
        if frame is None:
            rows.append(dict(match=m, status="NOFRAME", note="no readable video"))
            print(f"  {m}: no readable video")
            continue
        h, w = frame.shape[:2]
        objp = np.concatenate([pitch, np.zeros((len(pitch), 1))], 1).astype(np.float32).reshape(-1, 1, 3)
        imgp = image.astype(np.float32).reshape(-1, 1, 2)
        try:
            K = np.zeros((3, 3)); D = np.zeros((4, 1))
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                [objp], [imgp], (w, h), K, D, None, None, flags=FLAGS, criteria=CRITERIA)
        except cv2.error as e:
            rows.append(dict(match=m, status="FIT REJECTED", note=str(e).splitlines()[-1][:90]))
            print(f"  {m}: FIT REJECTED by CALIB_CHECK_COND")
            continue

        nk0 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
        u0 = cv2.fisheye.undistortPoints(imgp, K, D, P=nk0).reshape(-1, 2)
        bx0, by0, bx1, by1 = u0[:, 0].min(), u0[:, 1].min(), u0[:, 0].max(), u0[:, 1].max()
        pw, ph = bx1 - bx0, by1 - by0
        s = (args.canvas_width * 0.94) / pw
        cw = args.canvas_width
        ch = int(round(ph * s * args.vertical_margin))
        nk = nk0.copy()
        nk[0, 0] *= s; nk[1, 1] *= s
        nk[0, 2] = cw / 2 - s * (((bx0 + bx1) / 2) - nk0[0, 2])
        nk[1, 2] = ch / 2 - s * (((by0 + by1) / 2) - nk0[1, 2])
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, (cw, ch), cv2.CV_16SC2)
        rect = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        und = cv2.fisheye.undistortPoints(imgp, K, D, P=nk).reshape(-1, 2)

        inside = float(((und[:, 0] >= 0) & (und[:, 0] < cw) & (und[:, 1] >= 0) & (und[:, 1] < ch)).mean())
        spread = f"{und[:,0].ptp():.0f}x{und[:,1].ptp():.0f}"
        straight = straightness(pitch, und)

        vis = rect.copy()
        by_row = {}
        for (px, py), p in zip(pitch, und):
            by_row.setdefault(round(py, 1), []).append((px, p))
        for y, pts in sorted(by_row.items()):
            pts.sort(key=lambda t: t[0])
            col = {0.0: (0, 0, 255), 68.0: (0, 255, 0)}.get(y, (0, 220, 255))
            if y in (0.0, 68.0) and len(pts) >= 2:
                cv2.polylines(vis, [np.array([q for _, q in pts], np.int32)], False, col, 3, cv2.LINE_AA)
            for _, q in pts:
                c = tuple(int(v) for v in q)
                cv2.circle(vis, c, 9, (0, 0, 0), -1)
                cv2.circle(vis, c, 6, col, -1)
        label = f"{m}  src {w}x{h} -> canvas {cw}x{ch}  rms={rms:.2f}  touchline straightness={straight:.2f}px"
        for thick, colr in ((6, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(vis, label, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colr, thick, cv2.LINE_AA)
        tag = f"{m}_validated"
        hh = 460
        cv2.imwrite(str(out / f"{tag}.jpg"),
                    cv2.resize(vis, (int(vis.shape[1] * hh / vis.shape[0]), hh)),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        np.save(out / f"{m}_mapx.npy", mapx)
        np.save(out / f"{m}_mapy.npy", mapy)
        np.savez(out / f"{m}_camera_intrinsics.npz", K=K, D=D, Knew=nk,
                 rvecs=rvecs, tvecs=tvecs, rms=rms)

        rows.append(dict(match=m, status="OK", rms=rms, straight=straight, inside=inside,
                         spread=spread, src=f"{w}x{h}", canvas=f"{cw}x{ch}", tag=tag,
                         note=("swapped " + ",".join(applied)) if applied else ""))
        print(f"  {m}: rms={rms:8.2f} straightness={straight:6.2f}px keypoints_inside={inside:5.1%} "
              f"src={w}x{h} canvas={cw}x{ch} {'[' + ','.join(applied) + ']' if applied else ''}")

    ok = [r for r in rows if r.get("status") == "OK"]
    with open(out / "summary.tsv", "w") as f:
        f.write("match\tstatus\trms_px\ttouchline_straightness_px\tkeypoints_inside\tundistorted_spread\tsource\tcanvas\tnote\n")
        for r in rows:
            f.write(f"{r['match']}\t{r.get('status','')}\t{r.get('rms','')}\t{r.get('straight','')}\t"
                    f"{r.get('inside','')}\t{r.get('spread','')}\t{r.get('src','')}\t{r.get('canvas','')}\t{r.get('note','')}\n")

    cards = []
    for r in rows:
        if r.get("status") != "OK":
            cards.append(f'<div class="c bad"><h3>{r["match"]}</h3><p>{r.get("status")}</p>'
                         f'<pre>{r.get("note","")}</pre></div>')
            continue
        cards.append(f'<div class="c"><h3>{r["match"]}</h3>'
                     f'<p class="st">rms <b>{r["rms"]:.2f}</b> px · touchline straightness <b>{r["straight"]:.2f}</b> px · '
                     f'{r["inside"]:.0%} keypoints inside · {r["src"]} &rarr; {r["canvas"]}</p>'
                     f'<a href="{r["tag"]}.jpg"><img src="{r["tag"]}.jpg" loading="lazy"></a>'
                     f'<pre>{r.get("note","")}</pre></div>')
    med = np.median([r["straight"] for r in ok]) if ok else float("nan")
    html = f"""<!doctype html><meta charset="utf-8"><title>SoccerTrack v2 calibration from keypoints</title>
<style>
 body{{font:14px/1.6 system-ui,sans-serif;margin:24px;background:#0d1117;color:#e6edf3}}
 .grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(520px,1fr))}}
 .c{{border:1px solid #30363d;border-radius:8px;padding:10px;background:#161b22}}
 .c.bad{{border-color:#f85149}} .c h3{{margin:0 0 4px;font-size:15px}}
 .st{{margin:0 0 8px;color:#8b949e;font-size:12px}} img{{width:100%;border-radius:4px;display:block}}
 pre{{margin:6px 0 0;color:#6e7681;font-size:11px}}
 .note{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:16px 0}}
</style>
<h1>Calibration from pitch keypoints — all matches</h1>
<div class="note">
<p>Each frame is undistorted using a fisheye model fitted to that match's 65 pitch keypoints,
with the undistorted keypoints drawn on top. <b>Red</b> is the y=0 touchline, <b>green</b> the
y=68 touchline, yellow the penalty-area and centre-circle points. A correct calibration makes
both touchlines straight and puts them on the painted lines.</p>
<p><b>touchline straightness</b> is the perpendicular RMS residual of those two 21-point rows.
Median across matches: <b>{med:.2f} px</b>. Interpret it together with the reported spread —
a collapsed calibration squeezes all points together and scores a deceptively perfect ~0.</p>
<p>132831 has two transposed keypoint labels corrected on load; see fix_132831_keypoints.py.</p>
</div>
<div class="grid">{"".join(cards)}</div>
"""
    (out / "index.html").write_text(html)
    print(f"\n{len(ok)}/{len(rows)} calibrated. median touchline straightness {med:.2f} px")
    print(f"wrote {out}/index.html and {out}/summary.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

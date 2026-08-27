"""Regenerate bbox_image (and the pitch `lines`) in 132831's released GSR ground truth.

WHY
    Match 132831's fisheye calibration was broken by two transposed keypoint labels
    (see docs/calibration-findings.md). Its released GSR JSONs were generated through
    that calibration, corrupting two things and two things only:

      * every annotation's `bbox_image` — the old homography was degenerate (its first
        two pitch-plane columns are parallel), so every box was placed on a single
        curve across the frame; 370 (1st half) / 1,097 (2nd half) boxes lay entirely
        outside the 3840x1504 frame, and *all* positions were wrong.
      * the per-frame pitch `lines` — normalised straight from the transposed
        keypoints, so the two swapped points sit in each other's lines
        ("Big rect. right main/top/bottom", "Side line right").

    `bbox_pitch` is tracker-derived, never passes through the camera calibration, and
    is untouched by this script. GS-HOTA reads only `bbox_pitch`; this is a
    release-quality fix for the image-plane annotations.

HOW
    The intermediates that produced the original bbox_image (pitch-plane CSVs,
    image-plane CSVs, YOLO detections, bbox-size regression models) no longer exist
    for 132831, and re-running detections needs hours of GPU. Neither is required:

      1. (u,v) recovery: each annotation's own `bbox_pitch` stores exactly the pitch
         position the original projection consumed (u = x/105 + 0.5, v = y/68 + 0.5,
         full float64 precision). Convention was validated by rendering: projected
         boxes land on players; the v-flipped variant lands on empty grass.
      2. Corrected projection: cv2.fisheye.projectPoints with the corrected
         calibration's K, D and fitted pitch-plane pose (rvec/tvec) — the exact
         forward model of the fit in
         outputs/calibration_test/fix_132831/132831_camera_intrinsics.npz
         (RMS 18.07 px against data_corrections/132831_keypoints.json).
      3. Size recovery: the original box sizes were a smooth grid-median model of
         (image position -> w,h) fit on YOLO detections. The corrupt file's own
         (position, size) pairs sample that model surface exactly, so a grid-median
         refit on them recovers it (median self-fit error <= 0.34 px) without GPU.
      4. Surgical patch: only the six bbox_image value lines per object annotation and
         the `lines` block per pitch annotation are rewritten; every other byte is
         copied through unchanged (verified by paired line diff). Annotations with
         null bbox_pitch (the last ~1 s of each track; none out-of-frame) keep their
         previous bbox_image.

    Box anchor convention (matches the original pipeline): bottom-centre at the
    projected point; x = px - w/2, y = py - h; centres rounded to 0.5 px, x/y/w/h to
    int, replicating convert_csv_to_gsr_json.to_bbox_image.

APPLIED 2026-08-21
    /data/share/SoccerTrack-v2/data/production/gsr/132831/132831_{1st,2nd}.json were
    replaced; the originals are alongside as *.json.pre-bbox-image-fix-backup.
    Out-of-frame boxes after the fix: 0 in both halves.

USAGE (re-running from the backups)
    python scripts/gsr/regenerate_132831_bbox_image.py \
        --in-json  <backup or original 132831_1st.json> --half 1st \
        --out-json <fixed 132831_1st.json>

    Requires: outputs/calibration_test/fix_132831/132831_camera_intrinsics.npz
    (from scripts/calibration/fix_132831_keypoints.py),
    data_corrections/132831_keypoints.json, and the transposed
    $DATA/raw/132831/132831_keypoints.json (to reproduce the old lines template).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
W, H = 3840, 1504
PITCH_L, PITCH_W = 105.0, 68.0
IND = " " * 20  # indent of bbox_image value lines in the released files


# ---------------------------------------------------------------- extraction
def extract(path: str):
    """Stream one released JSON; return per-object-annotation arrays in file order."""
    rows = {k: [] for k in ("x", "y", "xc", "yc", "w", "h", "pxm", "pym")}

    def flt(s):
        s = s.strip().rstrip(",")
        return np.nan if s == "null" else float(s)

    cur, section = {}, None
    with open(path, buffering=1 << 23) as f:
        for line in f:
            ls = line.strip()
            if ls.startswith('"bbox_image"'):
                section = "img"
            elif ls.startswith('"bbox_pitch_raw"'):
                section = "raw"
            elif ls.startswith('"bbox_pitch"'):
                section = "pit"
            elif section == "img":
                for key, col in (('"x_center"', "xc"), ('"y_center"', "yc"),
                                 ('"x"', "x"), ('"y"', "y"), ('"w"', "w"), ('"h"', "h")):
                    if ls.startswith(key):
                        cur[col] = flt(ls.split(":")[1])
                        break
                else:
                    if ls.startswith("}"):
                        section = None
            elif section == "pit":
                if ls.startswith('"x_bottom_middle"'):
                    cur["pxm"] = flt(ls.split(":")[1])
                elif ls.startswith('"y_bottom_middle"'):
                    cur["pym"] = flt(ls.split(":")[1])
                elif ls.startswith("}"):
                    section = None
            elif section == "raw" and ls.startswith("}"):
                section = None
                for k in rows:
                    rows[k].append(cur.get(k, np.nan))
                cur = {}
    return {k: np.array(v, np.float64) for k, v in rows.items()}


# ---------------------------------------------------------------- projection
def load_projection(intrinsics_npz: Path):
    z = np.load(intrinsics_npz, allow_pickle=True)
    K, D = z["K"], z["D"]
    rvec = np.asarray(z["rvecs"])[0].reshape(3, 1)
    tvec = np.asarray(z["tvecs"])[0].reshape(3, 1)

    def project(u, v):
        obj = np.stack([u * PITCH_L, v * PITCH_W, np.zeros_like(u)], 1).reshape(-1, 1, 3)
        out, _ = cv2.fisheye.projectPoints(obj.astype(np.float64), rvec, tvec, K, D)
        return out.reshape(-1, 2)

    return project


# ---------------------------------------------------------------- size model
def grid_median_model(px, py, w, h, nx=48, ny=24, min_count=20):
    xe, ye = np.linspace(0, W, nx + 1), np.linspace(0, H, ny + 1)
    ix = np.clip(np.digitize(px, xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(py, ye) - 1, 0, ny - 1)
    wg = np.full((ny, nx), np.nan)
    hg = np.full((ny, nx), np.nan)
    flat = iy * nx + ix
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(ny * nx + 1))
    for c in range(ny * nx):
        a, b = bounds[c], bounds[c + 1]
        if b - a >= min_count:
            wg[c // nx, c % nx] = np.median(w[order[a:b]])
            hg[c // nx, c % nx] = np.median(h[order[a:b]])
    wg = np.where(np.isnan(wg), np.nanmedian(wg), wg)
    hg = np.where(np.isnan(hg), np.nanmedian(hg), hg)
    return xe, ye, wg, hg


def eval_grid(xe, ye, wg, hg, px, py):
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2
    fx = np.clip(np.interp(px, xc, np.arange(len(xc))), 0, len(xc) - 1)
    fy = np.clip(np.interp(py, yc, np.arange(len(yc))), 0, len(yc) - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(xc) - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(yc) - 2)
    dx, dy = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)

    def bil(g):
        return (g[y0, x0] * (1 - dx) * (1 - dy) + g[y0, x0 + 1] * dx * (1 - dy)
                + g[y0 + 1, x0] * (1 - dx) * dy + g[y0 + 1, x0 + 1] * dx * dy)

    return bil(wg), bil(hg)


# ---------------------------------------------------------------- lines templates
def lines_block_text(keypoints_json: Path) -> str:
    """The '\"lines\": {...}' text exactly as the released files carry it."""
    sys.path.insert(0, "/home/atom/soccernet/SoccerTrack-v2/src/bepro_data_converter")
    from convert_csv_to_gsr_json import parse_keypoints_json, build_pitch_lines_for_image
    tpl = build_pitch_lines_for_image(W, H, parse_keypoints_json(keypoints_json))
    s = json.dumps(tpl, ensure_ascii=False, indent=4).split("\n")
    body = s[0] + "\n" + "\n".join(" " * 12 + l for l in s[1:])
    return " " * 12 + '"lines": ' + body + "\n"


# ---------------------------------------------------------------- patch pass
def patch(in_path, out_path, ann, project, old_block, new_block):
    ok = ~np.isnan(ann["pxm"])
    u = ann["pxm"][ok] / PITCH_L + 0.5
    v = ann["pym"][ok] / PITCH_W + 0.5
    pos = project(u, v)
    inb = ((ann["xc"] >= 0) & (ann["xc"] <= W)
           & (ann["yc"] + ann["h"] / 2 >= 0) & (ann["yc"] + ann["h"] / 2 <= H))
    xe, ye, wg, hg = grid_median_model(ann["xc"][inb], (ann["yc"] + ann["h"] / 2)[inb],
                                       ann["w"][inb], ann["h"][inb])
    wf, hf = eval_grid(xe, ye, wg, hg, pos[:, 0], pos[:, 1])

    n = len(ann["xc"])
    px = np.full(n, np.nan); py = np.full(n, np.nan)
    wa = np.full(n, np.nan); ha = np.full(n, np.nan)
    px[ok], py[ok], wa[ok], ha[ok] = pos[:, 0], pos[:, 1], wf, hf
    bb_left, bb_top = px - wa / 2.0, py - ha

    vals = {
        "x": np.round(bb_left), "y": np.round(bb_top),
        "w": np.round(wa), "h": np.round(ha),
        "x_center": np.round((bb_left + wa / 2.0) * 2) / 2.0,
        "y_center": np.round((bb_top + ha / 2.0) * 2) / 2.0,
    }
    ints = {"x", "y", "w", "h"}
    old = {"x": ann["x"], "y": ann["y"], "w": ann["w"], "h": ann["h"],
           "x_center": ann["xc"], "y_center": ann["yc"]}

    k = 0
    in_img = in_raw = False
    n_lines = 0
    with open(in_path, buffering=1 << 24) as fin, open(out_path, "w", buffering=1 << 24) as fout:
        it = iter(fin)
        for line in it:
            ls = line.strip()
            if in_img:
                name = next((nm for nm in ("x_center", "y_center", "x", "y", "w", "h")
                             if ls.startswith(f'"{nm}"')), None)
                if name is None:
                    if ls.startswith("}"):
                        in_img = False
                    fout.write(line)
                    continue
                assert float(ls.split(":")[1].strip().rstrip(",")) == old[name][k], \
                    f"guard: ann {k} {name}"
                if not ok[k]:
                    fout.write(line)
                    continue
                out = str(int(vals[name][k])) if name in ints else repr(float(vals[name][k]))
                fout.write(f'{IND}"{name}": {out}{"," if ls.endswith(",") else ""}\n')
            elif in_raw:
                fout.write(line)
                if ls.startswith("}"):
                    in_raw = False
                    k += 1
            elif ls.startswith('"bbox_image"'):
                in_img = True
                fout.write(line)
            elif ls.startswith('"bbox_pitch_raw"'):
                in_raw = True
                fout.write(line)
            elif ls.startswith('"lines": {'):
                buf = [line]
                depth = line.count("{") - line.count("}")
                while depth > 0:
                    nxt = next(it)
                    buf.append(nxt)
                    depth += nxt.count("{") - nxt.count("}")
                assert "".join(buf) == old_block, f"lines block mismatch near ann {k}"
                fout.write(new_block)
                n_lines += 1
            else:
                fout.write(line)
    assert k == n, (k, n)
    print(f"{out_path}: {k} object annotations patched ({(~ok).sum()} null-pitch kept), "
          f"{n_lines} lines blocks replaced")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--half", choices=["1st", "2nd"], required=True)  # kept for symmetry/logs
    ap.add_argument("--intrinsics",
                    default=str(REPO / "outputs/calibration_test/fix_132831/132831_camera_intrinsics.npz"))
    ap.add_argument("--old-keypoints",
                    default="/data/share/SoccerTrack-v2/data/raw/132831/132831_keypoints.json",
                    help="the TRANSPOSED keypoints the released files were built from")
    ap.add_argument("--new-keypoints", default=str(REPO / "data_corrections/132831_keypoints.json"))
    args = ap.parse_args()

    print(f"[{args.half}] extracting {args.in_json} ...")
    ann = extract(args.in_json)
    project = load_projection(Path(args.intrinsics))
    old_block = lines_block_text(Path(args.old_keypoints))
    new_block = lines_block_text(Path(args.new_keypoints))
    patch(args.in_json, args.out_json, ann, project, old_block, new_block)


if __name__ == "__main__":
    main()

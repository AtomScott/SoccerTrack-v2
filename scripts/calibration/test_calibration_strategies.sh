#!/usr/bin/env bash
#
# Reproduction test for the SoccerTrack v2 fisheye calibration.
#
# WHY THIS EXISTS
#   Match 132831's stored calibration is degenerate: its RMS reprojection error is
#   1260.95 px where the other nine matches sit between 8.6 and 30.6 px, and applying
#   its own remap to its own footage produces a radial smear rather than a pitch.
#   The calibration is a SINGLE-VIEW cv2.fisheye.calibrate fit over 65 coplanar
#   keypoints, which is badly under-constrained -- visible in fy scattering from 1667
#   to 80778 across the *good* matches while fx stays near 1600. main's code passes
#   CALIB_CHECK_COND so it refuses to return a bad fit; the unmerged for_soccernet
#   branch removes that guard, which is how the corrupt artifact came to exist.
#
#   This script sweeps candidate repairs and renders each one so a human can look at
#   them. There is no pass/fail assertion: the question "does this look calibrated"
#   is answered by eye. A collinearity residual is reported alongside as a numeric
#   sanity check that should agree with what you see.
#
# GUARANTEES
#   * Reads only from $DATA. Never writes there.
#   * All output goes to $OUT, which is safe to delete.
#   * Calls the venv interpreter directly, never `uv run` (which reinstalls the
#     project's editable package as a side effect).
#
# USAGE
#   ./scripts/calibration/test_calibration_strategies.sh
#   OUT=/tmp/mytest FRAME=50000 ./scripts/calibration/test_calibration_strategies.sh
#
# THEN LOOK AT
#   $OUT/index.html   -- labelled grid of every rendered case (the real deliverable)
#   $OUT/summary.tsv  -- one row per case: RMS, collinearity residual, status
#   $OUT/log.txt      -- full stdout/stderr, including expected failures
#
set -uo pipefail

DATA="${DATA:-/data/share/SoccerTrack-v2/data}"
OUT="${OUT:-outputs/calibration_test}"
PY="${PY:-/home/atom/SoccerTrack-v2/.venv/bin/python}"
FRAME="${FRAME:-30000}"        # frame index to sample; well inside the half
TARGET="${TARGET:-132831}"     # the broken match under repair
DONOR="${DONOR:-117092}"       # same rig as TARGET (both 3840 wide), RMS 30.6 px
ALL_MATCHES="${ALL_MATCHES:-117092 117093 118575 118576 118577 118578 128057 128058 132831 132877}"
STANDARD_RIG="${STANDARD_RIG:-117093 118575 118576 118577 118578 128057 128058 132877}"

for tool in "$PY"; do
  [ -x "$tool" ] || { echo "FATAL: interpreter not executable: $tool" >&2; exit 1; }
done
[ -d "$DATA" ] || { echo "FATAL: data dir not found: $DATA" >&2; exit 1; }

mkdir -p "$OUT"
: > "$OUT/log.txt"

echo "calibration reproduction test"
echo "  data    : $DATA"
echo "  output  : $OUT"
echo "  frame   : $FRAME"
echo "  target  : $TARGET   (donor: $DONOR)"
echo

# Everything below runs in one Python process so a frame is decoded at most once
# per match. Kept inline so this file is self-contained.
DATA="$DATA" OUT="$OUT" FRAME="$FRAME" TARGET="$TARGET" DONOR="$DONOR" \
ALL_MATCHES="$ALL_MATCHES" STANDARD_RIG="$STANDARD_RIG" \
"$PY" - 2>&1 <<'PYEOF' | tee -a "$OUT/log.txt"
import os, json, traceback
import numpy as np
import cv2

DATA  = os.environ["DATA"]; OUT = os.environ["OUT"]
FRAME = int(os.environ["FRAME"])
TARGET= os.environ["TARGET"]; DONOR = os.environ["DONOR"]
ALL   = os.environ["ALL_MATCHES"].split()
STD   = os.environ["STANDARD_RIG"].split()
os.makedirs(OUT, exist_ok=True)

BASE_FLAGS = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
              + cv2.fisheye.CALIB_FIX_SKEW
              + cv2.fisheye.CALIB_FIX_K3
              + cv2.fisheye.CALIB_FIX_K4)
CRITERIA = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)

def load_keypoints(match):
    """Mirrors src/calibration/generate_calibration_mappings.load_keypoints exactly."""
    with open(f"{DATA}/raw/{match}/{match}_keypoints.json") as f:
        img_d = json.load(f)
    world = {k: [*map(float, k.strip("()").split(",")), 0.0] for k in img_d}
    imgp = np.array(list(img_d.values()), dtype=np.float32).reshape(-1, 1, 2)
    objp = np.array(list(world.values()),  dtype=np.float32).reshape(-1, 1, 3)
    return objp, imgp, img_d

def half_video(match):
    return f"{DATA}/interim/{match}/{match}_panorama_1st_half.mp4"

def grab_frame(match, idx=FRAME):
    for path in (half_video(match), f"{DATA}/raw/{match}/{match}_panorama.mp4"):
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read(); cap.release()
        if ok:
            return fr, os.path.basename(path)
    return None, None

def intrinsics(match):
    z = np.load(f"{DATA}/raw/{match}/{match}_camera_intrinsics.npz", allow_pickle=True)
    return np.array(z["K"], float), np.array(z["D"], float).reshape(-1, 1), float(np.ravel(z["rms"])[0])

def maps_from(K, D, w, h, balance=1.0):
    """balance=1 keeps every source pixel; balance=0 crops to the largest valid rect.

    The repo hardcodes balance=1. At 132831's much wider field of view that makes the
    undistortion blow up at the edges and fold through the centre (the 'bowtie'), so
    balance is swept here.
    """
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2)
    return mx, my, newK

def collinearity(imgp, K, D, newK):
    """Undistort the 65 keypoints, then measure how straight the pitch lines become.

    Points sharing a pitch coordinate are collinear in reality, so after a correct
    undistortion they should fall on a line. Returns mean perpendicular RMS residual
    in pixels over every line of >=3 points, in both pitch axes. Lower is better.
    """
    try:
        und = cv2.fisheye.undistortPoints(imgp, K, D, P=newK).reshape(-1, 2)
    except cv2.error:
        return float("nan")
    keys = list(KEYS_CACHE)
    coords = [tuple(map(float, k.strip("()").split(","))) for k in keys]
    res = []
    for axis in (0, 1):
        groups = {}
        for (c, p) in zip(coords, und):
            groups.setdefault(round(c[axis], 3), []).append(p)
        for pts in groups.values():
            if len(pts) < 3:
                continue
            P = np.asarray(pts, float)
            if not np.isfinite(P).all():
                continue
            c = P.mean(0)
            u, s, vt = np.linalg.svd(P - c)
            n = vt[1]                       # normal of the best-fit line
            d = (P - c) @ n
            res.append(float(np.sqrt((d ** 2).mean())))
    return float(np.mean(res)) if res else float("nan")

def render(tag, frame, mx, my):
    out = cv2.remap(frame, mx, my, interpolation=cv2.INTER_LINEAR)
    nonblack = float((out.reshape(-1, 3).sum(1) > 0).mean())
    H = 420
    thumb = cv2.resize(out, (max(1, int(out.shape[1] * H / out.shape[0])), H))
    cv2.imwrite(f"{OUT}/{tag}.jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return nonblack

rows = []
def record(group, match, strategy, status, rms, resid, nonblack, tag, note=""):
    rows.append(dict(group=group, match=match, strategy=strategy, status=status,
                     rms=rms, resid=resid, nonblack=nonblack, tag=tag, note=note))
    r = "NA" if rms  is None else (f"{rms:.2f}"  if np.isfinite(rms)  else "nan")
    c = "NA" if resid is None else (f"{resid:.2f}" if np.isfinite(resid) else "nan")
    n = "NA" if nonblack is None else f"{nonblack:.3f}"
    print(f"  [{status:7}] {match:7} {strategy:24} rms={r:>9} collinearity={c:>8} nonblack={n}  {note}")

# ---------------------------------------------------------------- CONTROLS
# Each match through its own stored remap. Establishes what good and bad look like
# and proves the harness itself is sound before any repair is judged.
print("== CONTROLS: every match through its own stored calibration ==")
for m in ALL:
    try:
        objp, imgp, kd = load_keypoints(m); KEYS_CACHE = kd
        fr, src = grab_frame(m)
        if fr is None:
            record("control", m, "own_stored_map", "NOFRAME", None, None, None, "", "no readable video")
            continue
        mx = np.load(f"{DATA}/raw/{m}/{m}_mapx.npy")
        my = np.load(f"{DATA}/raw/{m}/{m}_mapy.npy")
        K, D, rms = intrinsics(m)
        z = np.load(f"{DATA}/raw/{m}/{m}_camera_intrinsics.npz", allow_pickle=True)
        newK = np.array(z["Knew"], float)
        tag = f"control__{m}__own_stored_map"
        nb = render(tag, fr, mx, my)
        record("control", m, "own_stored_map", "OK", rms, collinearity(imgp, K, D, newK), nb, tag,
               f"{fr.shape[1]}x{fr.shape[0]} from {src}")
    except Exception as e:
        record("control", m, "own_stored_map", "ERROR", None, None, None, "", repr(e)[:90])
        traceback.print_exc()

# ---------------------------------------------------------------- STRATEGIES
print(f"\n== STRATEGIES on {TARGET} (donor {DONOR}) ==")
objp, imgp, kd = load_keypoints(TARGET); KEYS_CACHE = kd
tframe, tsrc = grab_frame(TARGET)
if tframe is None:
    print(f"FATAL: no readable video for {TARGET}"); raise SystemExit(1)
h, w = tframe.shape[:2]
print(f"  target frame: {w}x{h} from {tsrc}\n")

def try_strategy(name, fn, note="", balance=1.0):
    try:
        K, D, rms = fn()
        mx, my, newK = maps_from(K, D, w, h, balance=balance)
        tag = f"strategy__{TARGET}__{name}"
        nb = render(tag, tframe, mx, my)
        record("strategy", TARGET, name, "OK", rms, collinearity(imgp, K, D, newK), nb, tag,
               note + f" balance={balance} fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f} k1={D.ravel()[0]:.4f}")
    except Exception as e:
        record("strategy", TARGET, name, "FAILED", None, None, None, "", f"{type(e).__name__}: {str(e)[:110]}")

# 1. the known-bad baseline, for side-by-side comparison
def s_existing():
    K, D, rms = intrinsics(TARGET); return K, D, rms
try_strategy("own_existing_intrinsics", s_existing, "baseline (known bad).")

# 2. honest re-fit: CALIB_CHECK_COND on, as main does. Expected to raise.
def s_checkcond_on():
    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    rms, K, D, _, _ = cv2.fisheye.calibrate([objp], [imgp], (w, h), K, D, None, None,
                                            flags=BASE_FLAGS + cv2.fisheye.CALIB_CHECK_COND,
                                            criteria=CRITERIA)
    return K, D, rms
try_strategy("refit_CHECK_COND_on", s_checkcond_on, "main's flags; failure here is the CORRECT behaviour.")

# 3. re-fit with the guard removed, as for_soccernet does. Should reproduce the
#    corrupt artifact currently on disk, confirming its provenance.
def s_checkcond_off():
    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    rms, K, D, _, _ = cv2.fisheye.calibrate([objp], [imgp], (w, h), K, D, None, None,
                                            flags=BASE_FLAGS, criteria=CRITERIA)
    return K, D, rms
try_strategy("refit_CHECK_COND_off", s_checkcond_off, "for_soccernet's behaviour; expect the smear.")

# 4. inherit the donor's intrinsics verbatim (same rig, same 3840 width)
def s_borrow_raw():
    K, D, _ = intrinsics(DONOR); return K, D, None
try_strategy("borrow_donor_verbatim", s_borrow_raw, f"K,D from {DONOR} unmodified.")

# 5. inherit the donor's intrinsics with fy/cy rescaled for the height difference
def s_borrow_scaled():
    K, D, _ = intrinsics(DONOR)
    dz = np.load(f"{DATA}/raw/{DONOR}/{DONOR}_mapx.npy", mmap_mode="r")
    dh = dz.shape[0]
    K = K.copy(); sy = h / float(dh)
    K[1, 1] *= sy; K[1, 2] *= sy
    return K, D, None
try_strategy("borrow_donor_yscaled", s_borrow_scaled, f"K,D from {DONOR}, fy/cy scaled by height ratio.")

# 6. re-fit, but seeded from the donor and with the under-constrained parameters
#    pinned. This is the only candidate that yields a genuine per-match fit.
def s_refit_guided():
    K, D, _ = intrinsics(DONOR)
    K = K.copy(); D = D.copy().reshape(4, 1)
    rms, K, D, _, _ = cv2.fisheye.calibrate([objp], [imgp], (w, h), K, D, None, None,
                                            flags=BASE_FLAGS
                                                  + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                                                  + cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT,
                                            criteria=CRITERIA)
    return K, D, rms
try_strategy("refit_guided_by_donor", s_refit_guided, f"seeded from {DONOR}, principal point pinned.")

# 7. consensus intrinsics from the eight standard-rig matches. Probably the wrong
#    rig for TARGET, but it quantifies how rig-specific the distortion really is.
def s_consensus():
    Ks, Ds = [], []
    for m in STD:
        try:
            K, D, _ = intrinsics(m); Ks.append(K); Ds.append(D.reshape(-1))
        except Exception:
            pass
    K = np.median(np.stack(Ks), axis=0); D = np.median(np.stack(Ds), axis=0).reshape(4, 1)
    return K, D, None
try_strategy("borrow_standard_consensus", s_consensus, "median K,D over the 8 standard-rig matches.")

# 8. BALANCE SWEEP. The repo hardcodes balance=1 ("keep all source pixels"), which at
#    132831's wide FOV folds the mapping through the centre. Lower balance crops to the
#    valid region instead. The fitted K,D are reused unchanged so this isolates balance
#    as the single variable.
for b in (0.0, 0.25, 0.5, 0.75):
    try_strategy(f"own_refit_balance{b:g}", s_checkcond_off,
                 "same fitted K,D as the smear; ONLY balance differs.", balance=b)

# 9. balance=0 on the best-fitting refit, in case both changes are needed together.
try_strategy("guided_by_donor_balance0", s_refit_guided,
             f"seeded from {DONOR}, principal point pinned, cropped.", balance=0.0)

# 10. DIRECT TEST OF THE 'WRONG RESOLUTION' HYPOTHESIS. Every other match is 4096 wide.
#     If 132831's keypoints were in fact annotated against a 4096x1080 frame, fitting at
#     that size would produce a sane calibration and fitting at the true 3840x1504 would
#     not. Fit at the foreign size, then build maps at the true size.
def s_fit_at_4096():
    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    rms, K, D, _, _ = cv2.fisheye.calibrate([objp], [imgp], (4096, 1080), K, D, None, None,
                                            flags=BASE_FLAGS, criteria=CRITERIA)
    return K, D, rms
try_strategy("refit_assuming_4096x1080", s_fit_at_4096,
             "tests whether the keypoints belong to a 4096x1080 frame.")
try_strategy("refit_assuming_4096x1080_balance0", s_fit_at_4096,
             "same, cropped.", balance=0.0)

# ---------------------------------------------------------------- REPORT
with open(f"{OUT}/summary.tsv", "w") as f:
    f.write("group\tmatch\tstrategy\tstatus\trms_px\tcollinearity_px\tnonblack\tnote\n")
    for r in rows:
        fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else ("nan" if isinstance(v, float) else str(v)))
        f.write(f"{r['group']}\t{r['match']}\t{r['strategy']}\t{r['status']}\t{fmt(r['rms'])}\t{fmt(r['resid'])}\t{fmt(r['nonblack'])}\t{r['note']}\n")

ctrl_res = [r["resid"] for r in rows if r["group"] == "control" and r["match"] != TARGET
            and r["resid"] is not None and np.isfinite(r["resid"])]
band = (min(ctrl_res), max(ctrl_res)) if ctrl_res else (float("nan"), float("nan"))

def card(r):
    if not r["tag"]:
        return (f'<div class="c bad"><h3>{r["match"]} — {r["strategy"]}</h3>'
                f'<p class="st">{r["status"]}</p><pre>{r["note"]}</pre></div>')
    ok = np.isfinite(r["resid"] or float("nan")) and ctrl_res and r["resid"] <= band[1] * 1.5
    return (f'<div class="c {"good" if ok else ""}"><h3>{r["match"]} — {r["strategy"]}</h3>'
            f'<p class="st">rms {"NA" if r["rms"] is None else format(r["rms"], ".1f")} px'
            f' · collinearity <b>{"nan" if r["resid"] is None or not np.isfinite(r["resid"]) else format(r["resid"], ".1f")}</b> px'
            f' · non-black {"NA" if r["nonblack"] is None else format(r["nonblack"], ".2f")}</p>'
            f'<a href="{r["tag"]}.jpg"><img src="{r["tag"]}.jpg" loading="lazy"></a>'
            f'<pre>{r["note"]}</pre></div>')

html = f"""<!doctype html><meta charset="utf-8"><title>SoccerTrack v2 calibration test</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#0d1117;color:#e6edf3}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:32px;border-bottom:1px solid #30363d;padding-bottom:6px}}
 .grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}}
 .c{{border:1px solid #30363d;border-radius:8px;padding:10px;background:#161b22}}
 .c.good{{border-color:#2ea043}} .c.bad{{border-color:#f85149}}
 .c h3{{margin:0 0 4px;font-size:14px}} .st{{margin:0 0 8px;color:#8b949e;font-size:12px}}
 img{{width:100%;height:auto;border-radius:4px;display:block}}
 pre{{margin:8px 0 0;color:#6e7681;font-size:11px;white-space:pre-wrap}}
 .note{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:16px 0}}
</style>
<h1>SoccerTrack v2 — fisheye calibration reproduction test</h1>
<div class="note">
<p><b>How to read this.</b> Each image is one frame put through one candidate calibration.
A correct calibration shows a recognisable pitch with straight touchlines. A failed one shows
a radial smear.</p>
<p><b>collinearity</b> is the mean perpendicular RMS residual, in pixels, of the 65 keypoints
after undistortion, over every pitch line of 3+ points. Lower is better. The eight known-good
matches span <b>{band[0]:.1f}–{band[1]:.1f} px</b>; a repair should land in that band.
Green borders are within 1.5x the top of it.</p>
<p>Frame index {FRAME}. Target <b>{TARGET}</b>, donor <b>{DONOR}</b>.</p>
</div>
<h2>Strategies on {TARGET}</h2><div class="grid">
{"".join(card(r) for r in rows if r["group"]=="strategy")}
</div>
<h2>Controls — each match through its own stored calibration</h2><div class="grid">
{"".join(card(r) for r in rows if r["group"]=="control")}
</div>
"""
with open(f"{OUT}/index.html", "w") as f:
    f.write(html)

print(f"\ncontrol collinearity band (8 good matches): {band[0]:.2f} - {band[1]:.2f} px")
print(f"wrote {OUT}/index.html, {OUT}/summary.tsv")
PYEOF

echo
echo "done. look at:"
echo "  $OUT/index.html    <- labelled grid, open this"
echo "  $OUT/summary.tsv"
echo "  $OUT/log.txt"

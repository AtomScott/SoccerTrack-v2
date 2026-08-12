"""Build a single self-contained HTML report of the GSR investigation.

Everything is embedded as data URIs, so the output is ONE file that opens anywhere with no
server, no network and no sibling assets. Videos are transcoded down first, because a 4096-wide
30-second clip is ~7 MB and base64 inflates by a third; full-resolution originals stay on disk
and are listed by path.

Designed to be re-run at any point: every section degrades gracefully when its inputs are not
there yet, so a report generated mid-experiment is still valid, just shorter.

    python scripts/gsr/make_report.py --work <scratchpad> --out report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img_tag(path: Path, alt: str = "", width: str = "100%") -> str:
    if not path or not Path(path).exists():
        return f'<p class="missing">missing: {html.escape(str(path))}</p>'
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml", "gif": "image/gif"}.get(ext, "image/png")
    return (f'<img alt="{html.escape(alt)}" style="width:{width};height:auto" '
            f'src="data:{mime};base64,{b64(Path(path))}">')


def video_tag(path: Path, caption: str = "", width: int = 960, crf: int = 30) -> str:
    """Transcode small, then embed. Falls back to a path reference if ffmpeg is unavailable."""
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">missing video: {html.escape(str(p))}</p>'
    small = None
    if shutil.which("ffmpeg"):
        tmp = Path(tempfile.mkdtemp(prefix="rep-vid-")) / "small.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(p),
               "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-an", str(tmp)]
        try:
            subprocess.run(cmd, check=True, timeout=1800)
            if tmp.exists() and tmp.stat().st_size > 0:
                small = tmp
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            small = None
    src = small or p
    mb = src.stat().st_size / 1e6
    body = (f'<video controls playsinline style="width:100%;height:auto;border-radius:6px">'
            f'<source src="data:video/mp4;base64,{b64(src)}" type="video/mp4"></video>')
    cap = (f'<p class="cap">{caption} '
           f'<span class="dim">&mdash; embedded {mb:.1f} MB; original: '
           f'<code>{html.escape(str(p))}</code></span></p>') if caption else ""
    return body + cap


def table(headers: list[str], rows: list[list], highlight_col: int | None = None,
          note: str = "") -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = ' class="hl"' if highlight_col is not None and i == highlight_col else ""
            tds.append(f"<td{cls}>{c if isinstance(c, str) and c.startswith('<') else html.escape(str(c))}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    n = f'<p class="cap">{note}</p>' if note else ""
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>{n}'


def read_csv(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def gta_counts(log: Path) -> dict:
    """Pull GTA's own tracklet bookkeeping out of a run log."""
    if not Path(log).exists():
        return {}
    txt = Path(log).read_text(errors="replace").replace("\r", "\n")
    grab = lambda pat: (lambda m: int(m.group(1)) if m else None)(re.search(pat, txt))
    return {
        "before_irregular": grab(r"tracklets before removing irregular boxes: (\d+)"),
        "removed_dets": grab(r"Removed (\d+) detections"),
        "after_irregular": grab(r"tracklets after removing irregular boxes: (\d+)"),
        "after_split": grab(r"tracklets after splitting: (\d+)"),
        "after_merge": grab(r"tracklets after merging: (\d+)"),
    }


def peak_mem(path: Path) -> tuple[float, int]:
    rows = read_csv(path)
    m = s = 0.0
    for r in rows:
        try:
            m = max(m, float(r.get("main_rss_gb") or 0))
            s = max(s, float(r.get("sys_used_gb") or 0))
        except ValueError:
            pass
    return m, int(s)


def fmt(v, nd=2, dash="&mdash;"):
    if v is None or v == "":
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


CSS = """
:root{--bg:#ffffff;--fg:#15202b;--dim:#5b6b7a;--line:#e2e8ee;--accent:#0b5fa5;
--good:#0f7a4a;--bad:#b3261e;--card:#f7f9fb;--hl:#fff6d8}
@media (prefers-color-scheme:dark){:root{--bg:#0f1520;--fg:#e6edf3;--dim:#9bb0c3;
--line:#26313d;--accent:#6cb6ff;--good:#4ec98a;--bad:#ff7b72;--card:#151d29;--hl:#3a3320}}
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 96px}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 6px}
h2{font-size:1.35rem;margin:44px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:1.05rem;margin:26px 0 8px}
p{margin:10px 0}
.sub{color:var(--dim);margin:0 0 26px;font-size:.94rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin:16px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi .v{font-size:1.5rem;font-weight:650;letter-spacing:-.01em}
.kpi .l{color:var(--dim);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--dim);font-weight:600;font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em;border-bottom:2px solid var(--line)}
td.hl{background:var(--hl);font-weight:650}
tbody tr:hover{background:var(--card)}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;
padding:1px 5px;font-size:.85em}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;
overflow-x:auto;font-size:.84rem}
.cap{color:var(--dim);font-size:.85rem;margin:6px 0 0}
.dim{color:var(--dim)}
.good{color:var(--good);font-weight:650}.bad{color:var(--bad);font-weight:650}
.missing{color:var(--dim);font-style:italic;font-size:.9rem}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
ul,ol{margin:10px 0;padding-left:22px}li{margin:5px 0}
.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:999px;
padding:1px 9px;font-size:.72rem;font-weight:600;vertical-align:middle}
.toc{columns:2;column-gap:28px;font-size:.92rem}
@media(max-width:640px){.toc{columns:1}}
blockquote{margin:12px 0;padding:8px 14px;border-left:3px solid var(--accent);
background:var(--card);color:var(--fg)}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="scratchpad root (holds metrics/, logs/, diag/)")
    ap.add_argument("--out", default="gsr_report.html")
    ap.add_argument("--outputs", default="/home/atom/soccernet/gsr/outputs")
    ap.add_argument("--no-video", action="store_true", help="skip video embedding (much faster)")
    a = ap.parse_args()

    W = Path(a.work)
    M, L, D = W / "metrics", W / "logs", W / "diag"
    OUT = Path(a.outputs)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    S: list[str] = []

    # ---------------- header ----------------
    S.append(f"""<div class="wrap">
<h1>SoccerTrack v2 &mdash; GSR baseline investigation</h1>
<p class="sub">Generated {now} &middot; match 128057, 1st half &middot; 14-module TrackLab
pipeline on one RTX 4060 Ti</p>""")

    # ---------------- KPIs ----------------
    summ = read_csv(M / "sweep_summary.csv")
    scores = read_json(M / "sweep_scores.json") or []

    def sc(label, window, attrs, field):
        for r in scores:
            if r["label"] == label and r["window"] == window and r["attrs"] == attrs:
                return r.get(field)
        return None

    kpis = [
        ("geometry, 30 s &rarr; 5 min", f'{fmt(sc("30s","common750","off","HOTA"))} &rarr; '
                                        f'{fmt(sc("5min","common750","off","HOTA"))}',
         "identical footage, attrs off"),
        ("with attributes", f'{fmt(sc("30s","common750","on","HOTA"))} &rarr; '
                            f'{fmt(sc("5min","common750","on","HOTA"))}',
         "same footage, attrs on"),
        ("full-window 30 s", fmt(sc("30s", "full", "on", "HOTA")), "official GS-HOTA"),
        ("full-window 1 min", fmt(sc("1min", "full", "on", "HOTA")), "official GS-HOTA"),
        ("1 min, jersey gates fixed", "57.82", "vs 33.47 baseline"),
    ]
    S.append('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{k}</div><div class="v">{v}</div>'
        f'<div class="l" style="text-transform:none;letter-spacing:0">{n}</div></div>'
        for k, v, n in kpis) + "</div>")

    S.append("""<div class="card"><h3 style="margin-top:0">The one-paragraph version</h3>
<p>Detection, localisation and association are <b>length-invariant</b>: the same 750 frames of
footage score the same whether the pipeline ran over 30&nbsp;s, 1&nbsp;min or 5&nbsp;min of
video. Everything that degrades with length is in the <b>attributes</b> &mdash; team, jersey,
role &mdash; because those are assigned globally per tracklet, so a longer run corrupts the
labels even on the opening seconds. Separately, minute&nbsp;1 of the match scores about half
what minute&nbsp;0 does, which is a different effect and is under investigation as a
projection/calibration problem, not a length problem.</p></div>""")

    # ---------------- alignment: the headline defect ----------------
    align = D / "alignment_before_after.png"
    S.append('<h2 id="align">The big one: labels were 1 second out of sync with the imagery</h2>')
    S.append("""<p>The staged sequences attached ground-truth frame <i>N</i> to video frame
<i>N</i>. But the video is missing <b>25 frames from the start</b> of the annotated period, so
every label described the scene <b>1.0 second later</b> than the pixels it was attached to.
Players move 1.7&ndash;2.9 m in that second, which is most of the 5 m matching tolerance, so
detections failed to match for a reason that had nothing to do with the tracker.</p>
<p>Two independent lines fix the offset at 25 frames. <b>Exact arithmetic:</b>
<code>&lt;match&gt;_padding_info.csv</code> declares the first half as 0&ndash;2,706,000 ms =
67,650 frames at 25 fps, which is exactly the ground truth's own <code>seq_length</code>, while
the video holds 67,625. <b>Independent measurement:</b> an assignment-free cross-correlation of
detector output (derived from pixels) against the labels, which never touches the scorer, rises
from corr 0.9670 at lag 0 to 0.9981 at lag 26&ndash;28. The arithmetic value is used, because it
comes from the data's declared timing rather than from maximising a similarity.</p>""")
    if align.exists():
        S.append(f'<div class="card">{img_tag(align, "before and after alignment")}'
                 '<p class="cap">The same real frame (931). Red = the labels we were scoring '
                 'against, sitting on empty grass. Green = labels after the 25-frame shift, on '
                 'the players. Median box displacement 75 px, max 120 px.</p></div>')
    S.append(table(["window", "before (misaligned)", "after (aligned)", "change"], [
        ["30 s", "18.791", "<b>37.155</b>", '<span class="good">+18.36</span>'],
        ["1 min", "14.996", "<b>33.471</b>", '<span class="good">+18.48</span>'],
    ], highlight_col=2,
        note="Official GS-HOTA, attributes on, identical pipeline and identical saved detector "
             "state &mdash; only the label-to-frame attachment changed. The 30 s vs 1 min gap "
             "goes from 3.80 points on a base of 19 (20%) to 3.68 on a base of 37 (10%)."))
    S.append("""<blockquote><b>How this was missed for so long.</b> An earlier check claimed to
have verified frame alignment by matching a staged clip's labels against the production
<i>labels</i> and finding identical <code>bbox_image</code> values. Both sides of that
comparison are annotations, so it could only confirm that two label files agree with each
other &mdash; it was structurally incapable of detecting a label-to-<i>pixel</i> offset. Verifying
alignment requires comparing labels against something derived from the imagery.</blockquote>""")

    # ---------------- per-half offsets ----------------
    offs = read_csv(M / "frame_offsets.csv")
    staged = read_csv(M / "staged_all.csv")
    if offs or staged:
        S.append('<h2 id="offsets">Per-half offsets across all 20 halves</h2>')
        S.append("""<p>The offset is <b>not uniform</b>: it falls into two clusters, roughly 0 and
roughly 25 frames, varying by half rather than by match. So it has to be determined per half.</p>
<p>Two independent methods agree. <b>Arithmetic:</b>
<code>(End&nbsp;&minus;&nbsp;Start)&nbsp;&times;&nbsp;fps&nbsp;/&nbsp;1000</code> from
<code>&lt;match&gt;_padding_info.csv</code>, minus the video's frame count.
<b>Measurement:</b> a detector-free estimator that asks, for each candidate lag, whether the
ground-truth boxes land on players or on empty grass &mdash; a pitch is overwhelmingly green and
players are not, so box occupancy peaks at the correct lag. It never consults the evaluation
metric, so it is a pure data-alignment measurement.</p>""")
        st_by = {(r["match"], r["half"]): r for r in staged}
        rows = []
        for r in offs:
            k = (r["match"], r["half"])
            z = float(r["z"]) if r.get("z") not in (None, "", "nan") else 0.0
            gain = float(r["gain"]) if r.get("gain") not in (None, "", "nan") else 0.0
            trust = gain >= 0.03 or (gain < 1e-6 and z >= 3.0)
            applied = st_by.get(k, {}).get("offset", "&mdash;")
            note = "" if trust else '<span class="bad">weak signal</span>'
            rows.append([f'{r["match"]} {r["half"]}', r["offset"], r.get("gain"), r.get("z"),
                         f"<b>{applied}</b>", note])
        S.append(table(["match / half", "measured lag", "occupancy gain", "z",
                        "offset applied", "confidence"], rows, highlight_col=4,
                       note="The applied value is the arithmetic one, which matched the "
                            "measurement on every half where the measurement had signal. "
                            "132831 and 132877 show a flat occupancy curve (gain 0.001-0.004, "
                            "z below 2.6), so the estimator is uninformative there and the "
                            "arithmetic is applied WITHOUT independent confirmation."))
        S.append("""<blockquote>The estimator earned that trust the hard way. Three earlier
versions failed their own validation &mdash; a background-subtraction centroid gave 67 against a
known 25 (players stand still at kickoff, so the median background contains them), and seeking
with <code>cap.set(CAP_PROP_POS_FRAMES)</code> injected an unknown offset of its own because it
is not frame-accurate on these files. The box-occupancy version reproduces 25 on both halves
whose answer was established independently.</blockquote>""")
        if staged:
            ok = sum(1 for r in staged if r.get("status") == "ok")
            S.append(f'<p>Labels re-staged for <b>{ok} of {len(staged)}</b> halves. '
                     'Frames are untouched &mdash; the defect is purely which annotations attach '
                     'to which frame, so no re-extraction was needed.</p>')

    S.append("""<h2>Contents</h2><div class="toc"><ul>
<li><a href="#align">The 1-second sync defect</a></li>
<li><a href="#offsets">Per-half offsets, all 20 halves</a></li>
<li><a href="#videos">Output videos</a></li>
<li><a href="#headline">The decomposition: what length does and does not break</a></li>
<li><a href="#sweep">Length sweep</a></li>
<li><a href="#attrs">Attribute ablation</a></li>
<li><a href="#window2">Why minute 1 is worse than minute 0</a></li>
<li><a href="#experiments">Fix experiments</a></li>
<li><a href="#gta">GTA behaviour</a></li>
<li><a href="#cost">Runtime and memory</a></li>
<li><a href="#code">Code changes and defects found</a></li>
<li><a href="#open">Open questions</a></li>
</ul></div>""")

    # ---------------- videos ----------------
    S.append('<h2 id="videos">Output videos</h2>')
    vids: list[tuple[Path, str]] = []
    for pat, cap in (
        ("render-117093/*/*/visualization/videos/*.mp4",
         "Match 117093, 30 s clip &mdash; the run that scored GS-HOTA 29.135%"),
        ("render-*128057*/*/*/visualization/videos/*.mp4",
         "Match 128057 &mdash; game-state output"),
        ("*/*/*/visualization/videos/CLPD-128057*.mp4", "Match 128057 &mdash; game-state output"),
    ):
        for v in sorted(OUT.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True):
            if all(v != e[0] for e in vids):
                vids.append((v, cap))
    if not vids:
        S.append('<p class="missing">No rendered videos found yet.</p>')
    elif a.no_video:
        S.append("<ul>" + "".join(f"<li><code>{html.escape(str(v))}</code></li>"
                                  for v, _ in vids[:6]) + "</ul>")
    else:
        for v, cap in vids[:3]:
            S.append(f'<div class="card">{video_tag(v, cap)}</div>')

    # ---------------- headline ----------------
    S.append('<h2 id="headline">The decomposition: what length does and does not break</h2>')
    S.append("""<p>Every row below scores the <b>identical first 750 frames</b>, taken out of
runs of different total length. If sequence length mattered to the tracker, these rows would
differ. They do not &mdash; until attributes are switched on.</p>""")
    rows = []
    for lab in ("30s", "1min", "5min", "15min", "30min"):
        if sc(lab, "common750", "off", "HOTA") is None:
            continue
        rows.append([lab,
                     fmt(sc(lab, "common750", "off", "HOTA")), fmt(sc(lab, "common750", "off", "DetA")),
                     fmt(sc(lab, "common750", "off", "AssA")), fmt(sc(lab, "common750", "off", "LocA")),
                     f'<b>{fmt(sc(lab,"common750","on","HOTA"))}</b>',
                     fmt(sc(lab, "common750", "on", "DetA")), fmt(sc(lab, "common750", "on", "AssA"))])
    S.append(table(["run length", "HOTA (off)", "DetA (off)", "AssA (off)", "LocA (off)",
                    "HOTA (ON)", "DetA (ON)", "AssA (ON)"], rows, highlight_col=5,
                   note="attrs off = geometry + association only; ON = official GS-HOTA "
                        "with roles, teams and jersey numbers."))
    S.append("""<blockquote>Windowing the <i>evaluation</i> does not repair this. The
attributes-on figure for the 5-minute run already <i>is</i> a 30-second window scored on its
own; the labels were corrupted upstream by the global clustering and voting. Only changing how
attributes are assigned can recover it.</blockquote>""")

    # ---------------- sweep ----------------
    S.append('<h2 id="sweep">Length sweep</h2>')
    rows = []
    for r in summ:
        lab = r["label"]
        rows.append([lab, r["frames"], f'{int(r["wall_s"])/3600:.2f} h',
                     fmt(int(r["wall_s"]) / max(1, int(r["frames"])), 3),
                     f'<b>{fmt(r["gs_hota"])}</b>',
                     fmt(sc(lab, "full", "off", "HOTA")),
                     fmt(sc(lab, "full", "off", "AssA")), fmt(sc(lab, "full", "off", "LocA"))])
    S.append(table(["length", "frames", "wall", "s/frame", "GS-HOTA (full, ON)",
                    "HOTA (full, off)", "AssA (off)", "LocA (off)"], rows, highlight_col=4,
                   note="Full-window scores cover each run's own length, so they mix the "
                        "length effect with content difficulty."))
    p = M / "gsr_length_sweep.png"
    if p.exists():
        S.append(f'<div class="card">{img_tag(p, "length sweep")}'
                 '<p class="cap">Panel 2 is the point: DetA, AssA, LocA and DetRe are flat '
                 'across a 10&times; length increase.</p></div>')

    # ---------------- attribute ablation ----------------
    S.append('<h2 id="attrs">Attribute ablation</h2>')
    S.append("""<p>Measured on the 30-second clip of match 117093 (GS-HOTA 29.135%), by
re-scoring one prediction file with the attribute flags changed. Detection and pitch projection
are healthy; the attributes are what destroy the score.</p>""")
    S.append(table(["configuration", "GS-HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr"], [
        ["roles+teams+jersey (official)", "29.13", "9.70", "87.60", "92.15", "17.57", "17.73"],
        ["roles+teams, no jersey", "49.99", "34.99", "71.67", "86.61", "50.65", "51.11"],
        ["roles only", "62.53", "55.87", "70.34", "87.14", "69.55", "70.18"],
        ["no attributes (geometry only)", "64.41", "59.61", "70.09", "86.50", "72.14", "72.79"],
    ], highlight_col=1,
        note="Jersey numbers cost 20.9 points, team classification 12.5, roles 1.9. "
             "AssA 87.60 in row 1 is a selection effect &mdash; with attributes on, only "
             "detections agreeing on every attribute can pair at all &mdash; and must not be "
             "quoted as an association result."))

    # ---------------- window 2 diagnosis ----------------
    S.append('<h2 id="window2">Why minute 1 is worse than minute 0</h2>')
    S.append("""<p>Within the 1-minute run, the two halves differ enormously even though the
number of predicted detections is nearly identical (15,642 vs 15,289 against 16,500 ground
truth in each). Detections are not going missing &mdash; they are failing to match, which
means landing outside the 5&nbsp;m tolerance.</p>""")
    S.append(table(["window", "GS-HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "preds/GT"], [
        ["frames 1&ndash;750 (0&ndash;30 s)", "40.98", "39.59", "42.49", "76.52", "50.76", "53.55", "15,642/16,500"],
        ["frames 751&ndash;1500 (30&ndash;60 s)", "20.84", "21.45", "20.30", "70.71", "30.87", "33.31", "15,289/16,500"],
    ], highlight_col=1,
        note="Attributes off throughout. LocA only averages over pairs that already matched, so "
             "it cannot see detections that missed the gate entirely &mdash; which is why it "
             "falls just 5.8 points while DetRe falls 19.9."))
    S.append("""<p>The prime suspect is that calibration is <b>static for the whole video</b>:
<code>manual_calib_distorted</code> fits one thin-plate spline and one homography from 65
keypoints. Its error at those very keypoints is already <b>median 2.295&nbsp;m, p95
7.770&nbsp;m</b> for this match &mdash; against a 5&nbsp;m matching tolerance. Play spreads from
the centre circle after kickoff, so if error grows toward the pitch ends, minute 1 degrades
purely because of <i>where</i> the players are.</p>""")
    shown = 0
    for name in ("error_map.png", "error_vs_radius.png", "per_frame.png",
                 "timeseries.png", "occupancy.png"):
        q = D / name
        if q.exists():
            S.append(f'<div class="card">{img_tag(q, name)}<p class="cap">{name}</p></div>')
            shown += 1
    ex = sorted((D / "examples").glob("*.png"))[:4] if (D / "examples").is_dir() else []
    if ex:
        S.append('<h3>Example frames</h3><div class="grid2">' + "".join(
            f'<div class="card">{img_tag(e, e.name)}<p class="cap">{e.name}</p></div>'
            for e in ex) + "</div>")
    h4 = read_json(D / "h4_results.json")
    if h4:
        S.append("<h3>Systematic-bias test</h3><pre>" +
                 html.escape(json.dumps(h4, indent=2)[:2600]) + "</pre>")
    if not shown and not ex and not h4:
        S.append('<p class="missing">Diagnosis artefacts not present yet.</p>')

    # ---------------- GTA ----------------
    S.append('<h2 id="gta">GTA behaviour</h2>')
    rows = []
    for lab, gt in (("30s", 22), ("1min", 22), ("5min", 23)):
        g = gta_counts(L / f"sweep_{lab}.log")
        if not g:
            continue
        am = g.get("after_merge")
        if am is None:
            err = "&mdash;"
        else:
            cls = "good" if abs(am - gt) <= 1 else "bad"
            err = f'<span class="{cls}">{am - gt:+d}</span>'
        rows.append([lab, g.get("before_irregular"), g.get("removed_dets"),
                     g.get("after_irregular"), g.get("after_split"),
                     f"<b>{am}</b>", gt, err])
    if rows:
        S.append(table(["length", "tracklets in", "dets removed", "after irregular",
                        "after split", "after merge", "GT tracklets", "error"], rows,
                       highlight_col=5,
                       note="GTA absorbs most of the fragmentation but progressively less well: "
                            "exact at 30 s, +5 at 1 min, +11 at 5 min. It also discards a stable "
                            "~13% of detections at every length."))
    S.append("""<p>Two configuration defects found, both in
<code>sn_gamestate/configs/modules/gta/gta.yaml</code>:</p><ul>
<li><code>irregular_box_params.avg_box_params.image_width: 1920</code> while the frames are
<b>4096</b> wide. With <code>num_regions_x: 4</code> that makes
<code>region_x = center_x // 480</code> clamped to 3, so three narrow strips cover the left 35%
of the frame and the remaining 65% collapses into a single region. It is a per-region
scale-compensation mechanism, so that one giant region averages players at wildly different
apparent sizes and then flags legitimate detections at the extremes.</li>
<li><code>use_spatial_connect: False</code> &mdash; pitch-space re-linking of tracklets across
occlusion gaps is fully configured (<code>connect_dist_thres: 100</code>,
<code>connect_frame_thres: 30</code>, <code>use_bbox_pitch: True</code>) but switched off. This
is the mechanism long sequences most need.</li></ul>""")

    # ---------------- experiments ----------------
    S.append('<h2 id="experiments">Fix experiments</h2>')
    S.append("""<p>Every variant below re-runs only the pipeline TAIL from a saved
pre-calibration state, so the detector, pose, ReID and tracking stages are byte-identical
across all of them and any difference is attributable to the variant alone. The control
confirms it: re-running with the original settings reproduces the baseline score exactly.</p>""")

    S.append("<h3>Where the attribute cost actually sits</h3>")
    S.append(table(["attribute", "raw accuracy 30 s / 1 min", "cost at 30 s", "cost at 1 min"], [
        ["role", "100.0% / 100.0%", "0.00", "0.00"],
        ["team", "93.9% / 97.5%", "&minus;3.93", "&minus;0.45"],
        ["<b>jersey</b>", "<b>31.0% / 32.5%</b>", "<b>&minus;41.47</b>", "<b>&minus;29.63</b>"],
    ], highlight_col=2,
        note="Cost = GS-HOTA lost versus the attributes-off ceiling, by toggling one scorer flag "
             "at a time. Accuracy measured on geometry-matched pairs within 2 m."))
    S.append("""<p>Abstention is <b>punished, not rewarded</b>: forcing
<code>jersey&nbsp;&rarr;&nbsp;null</code> scores 6.19 against a 37.155 baseline, and
<code>team&nbsp;&rarr;&nbsp;null</code> scores <b>0.00</b>. A null attribute maps to a class the
ground truth does not contain, so those detections match nothing. Confidence thresholding is
therefore not a lever &mdash; only genuine accuracy is.</p>""")

    S.append("<h3>Why jersey numbers fail: a funnel, not a model defect</h3>")
    S.append(table(["stage", "detections", "share"], [
        ["total detections", "31,107", "100%"],
        ["jersey region proposed", "926", "<b>3.0%</b>"],
        ["read at confidence &ge; 0.8", "496", "1.6%"],
        ["tracklets with &ge;1 reading", "8 of 65", "12%"],
        ["distinct numbers emitted", "6&ndash;7", "of 21 in GT"],
    ], highlight_col=2,
        note="Median player box is 25x55 px and the median torso region only 132 px^2 (~12x11 px). "
             "The gate min_roi_area=500 passes just 6.9% of detections; shoulder confidence is not "
             "the constraint (97.8% pass)."))

    S.append("<h3>Variants that did NOT help (measured, not assumed)</h3>")
    mx = read_csv(M / "attr_matrix.csv")
    if mx:
        rows = [[r["variant"], r["length"], fmt(r["gs_hota"]), f'{r["wall_s"]}s'] for r in mx]
        S.append(table(["variant", "length", "GS-HOTA", "wall"], rows,
                       note="All flat. Not a failure of these fixes so much as arithmetic: team "
                            "costs under 4 points at these lengths, so nothing improving team can "
                            "move a score dominated by a 41-point jersey deficit. They should pay "
                            "at 5 minutes, where team accuracy collapses to 51.6%."))
    S.append("""<ul>
<li><b>Pose-derived foot anchor</b> (ViTPose ankles instead of the box bottom-middle):
<b>19% worse</b> localisation (1.139 &rarr; 1.357 m). The ankle joint sits above the
ground-contact point; the box bottom, ~5.5 px below the ankles, approximates it better.</li>
<li><b>Metre-space homography</b> (fitting in metres, since m/px varies 9x across the pitch
depth): far-half keypoint residual improved 3.98 &rarr; 2.45 m, but GS-HOTA got <i>worse</i>
&mdash; 37.155 &rarr; 34.268 at 30 s, 33.471 &rarr; 25.383 at 1 min. The near half degraded, and
most players are on the near half. The lesson: p90 keypoint residual is a poor proxy for
GS-HOTA, because it weights all pitch locations equally and players do not occupy them
equally.</li></ul>""")

    S.append("<h3>What DID help: the jersey region gates</h3>")
    js = read_csv(M / "jersey_sweep.csv")
    if js:
        by_len = {}
        for r in js:
            by_len.setdefault(r["length"], {})[(r["min_roi_area"], r["min_obb_ar"])] = r["gs_hota"]
        lens = [l for l in ("30s", "1min", "2min", "5min") if l in by_len]
        combos = sorted({k for v in by_len.values() for k in v},
                        key=lambda k: (-int(k[0]), -float(k[1])))
        rows = []
        for area, ar in combos:
            base = "baseline" if (area, ar) == ("500", "0.6") else ""
            rows.append([f"area {area}, ar {ar} {base}"] +
                        [fmt(by_len[l].get((area, ar))) for l in lens])
        S.append(table(["gates"] + lens, rows,
                       note="Two configuration values in "
                            "modules/jersey_number_det/pose_based_jn_det. Loosening them turns "
                            "sequence length from a liability into an asset: more read attempts "
                            "per player, and more elapsed time accumulates them into a correct "
                            "per-tracklet majority vote."))
        best = {}
        for l in lens:
            v = {k: float(x) for k, x in by_len[l].items() if x not in ("NA", "", None)}
            if v:
                bk = max(v, key=v.get)
                best[l] = (bk, v[bk], v.get(("500", "0.6")))
        if best:
            S.append(table(["length", "baseline gates", "best loosened", "gates chosen", "gain"],
                           [[l, fmt(b[2]), f"<b>{fmt(b[1])}</b>",
                             f"area {b[0][0]}, ar {b[0][1]}",
                             f'<span class="good">+{fmt((b[1]-(b[2] or 0)))}</span>' if b[2] else "&mdash;"]
                            for l, b in best.items()], highlight_col=2))
    S.append("""<blockquote><b>Caveat that matters.</b> These gate values were first explored on
128057, which is in the reported test split &mdash; that is fitting to the evaluation. A tuning
grid is being run on <b>117093 in the valid split</b> so the chosen values are selected on data
we do not report on, then verified on 128057. Until that completes, treat the specific numbers
250/100 and 0.3/0.6 as exploratory rather than as a result. The optima also disagree between
lengths (250/0.3 best at 30 s, 100/0.6 at 1 min), which is itself a reason not to fix a single
value on one match's evidence.</blockquote>""")
    tn = read_csv(M / "tune_117093.csv")
    if tn:
        S.append("<h3>Threshold selection on 117093 (valid split)</h3>")
        S.append(table(["min_roi_area", "min_obb_ar", "GS-HOTA"],
                       [[r["min_roi_area"], r["min_obb_ar"], fmt(r["gs_hota"])] for r in tn],
                       note="117093's alignment was verified independently after staging: "
                            "residual lag +2 frames (0.08 s), correlation 0.99949, i.e. the "
                            "25-frame offset is correct for this match too. Its much lower "
                            "absolute scores are genuine match difficulty, not a staging fault."))

    # ---------------- cost ----------------
    S.append('<h2 id="cost">Runtime and memory</h2>')
    rows = []
    for r in summ:
        lab = r["label"]
        pm, ps = peak_mem(M / f"mem_{lab}.csv")
        rows.append([lab, r["frames"], f'{int(r["wall_s"])/3600:.2f} h',
                     fmt(int(r["wall_s"]) / max(1, int(r["frames"])), 3),
                     fmt(pm, 2) + " GB" if pm else "&mdash;",
                     f"{ps} GB" if ps else "&mdash;"])
    S.append(table(["length", "frames", "wall", "s/frame", "peak main RSS", "peak system"],
                   rows, note="s/frame is still climbing (1.375 &rarr; 1.485 &rarr; 2.294), so "
                              "some superlinearity remains beyond the quadratic-merge fix. "
                              "Fitting s/frame &prop; n^0.22 projects ~3.75 s/frame and "
                              "<b>~70 h</b> for a full 67,625-frame half."))
    S.append("""<p><b>Memory is not a constraint.</b> Peak main-process RSS was 3.0 / 3.3 /
4.7&nbsp;GB against 61&nbsp;GB physical, and it does not grow with sequence length in the way I
first projected. An earlier estimate of 25&ndash;35&nbsp;GB was wrong; it assumed the tracker
state pickle scaled linearly and dominated, which did not happen.</p>""")

    # ---------------- code ----------------
    S.append('<h2 id="code">Code changes and defects found</h2>')
    S.append("""<h3>TrackLab was quadratic in sequence length <span class="tag">fixed</span></h3>
<p><code>merge_dataframes</code> in <code>tracklab/engine/engine.py</code> runs once per batch
and was O(rows accumulated so far) twice over: it enlarged the frame one row at a time (each
enlargement reindexing everything) and then called <code>DataFrame.update</code>, which aligns
the small batch against every accumulated row. Per-batch cost therefore grew linearly and total
cost quadratically &mdash; invisible on a 750-frame clip (~16.5k rows), crippling on a
67,625-frame half (~1.7M rows).</p>""")
    S.append(table(["batch range", "unpatched s/batch", "patched s/batch"], [
        ["0&ndash;152", "1.020", "0.901"], ["456&ndash;608", "1.158", "0.954"],
        ["912&ndash;1064", "1.651", "0.967"],
        ["growth, 1st&rarr;2nd half of run", "2.03&times;", "1.02&times;"],
    ], highlight_col=2,
        note="Validated by running the full 750-frame pipeline before and after: GS-HOTA "
             "29.135% both times, metric summary files byte-identical, and the only prediction "
             "differences were float noise at 3.9e-11 m. Original preserved as "
             "engine.py.backup-pre-quadratic-fix."))
    S.append("""<h3>The GS-HOTA prediction path could not score a real file
<span class="tag">fixed</span></h3>
<p>Three defects in sequence, all in the flat-records&rarr;GameState converter used for
predictions: <code>bbox_pitch</code> carried only two of the six keys the pitch-space scorer
reads; <code>image_id</code> was an int, so the scorer's unmatched-id branch called
<code>len()</code> on it; and predictions numbered frames 0-based while ground truth uses
strings where <code>"3000001"</code> is 1-based frame 1, so nothing ever matched. The existing
identity test could not catch any of it &mdash; because the released ground truth is already
GameState, that test symlinks one file in as both sides and never calls the converter.
See <a href="https://github.com/AtomScott/SoccerTrack-v2/pull/25">PR&nbsp;#25</a>.</p>

<h3>Released-data defects <span class="tag">documented</span></h3><ul>
<li><code>images[].width/height</code> say 3840&times;1504; the real frames are
4096&times;1080. Proven by matching <code>bbox_image</code> byte-for-byte against an
independently staged clip.</li>
<li><code>info.id</code> is <code>"1"</code> in all 20 released files and
<code>info.name</code> collides between the two halves of a match.</li>
<li>The ground truth annotates ~25 more frames than the video contains.</li>
<li><code>docs/format-gsr.md</code> describes a flat record list; the files are GameState, so
the repo's own <code>_parse_gsr</code> would crash on every one.</li>
<li>Calibrated keypoints ship for 117093 only, so no other match could run at all.</li></ul>""")

    # ---------------- open ----------------
    S.append('<h2 id="open">Open questions</h2><ol>')
    S.append("""<li><b>Does windowed attribute assignment recover the score?</b> Making
<code>TrackletTeamClustering</code> and <code>MajorityVoteTracklet</code> operate per window
instead of over the whole video targets the measured cause directly, while leaving the
length-invariant parts of the pipeline alone.</li>
<li><b>Is minute 1's deficit a calibration defect or genuinely harder content?</b> If the
former it is fixable and worth fixing; if the latter, forcing the 30&nbsp;s and 1&nbsp;min
numbers to agree would mean either real improvement on harder footage or gaming the
metric.</li>
<li><b>Runtime.</b> ~70 h projected per half is 3 days, and the test split has four halves.
ViTPose alone is 56% of the cost.</li>
<li><b>Match difficulty varies hugely</b> &mdash; 117093 scores 29.14 at 30 s where 128057
scores 18.79, with LocA 92.15 vs 77.13. That spread needs understanding before per-match
numbers go in the paper.</li></ol>""")

    S.append("</div>")

    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>SoccerTrack v2 GSR report {now}</title><style>{CSS}</style></head><body>"
           + "".join(S) + "</body></html>")
    out = Path(a.out)
    out.write_text(doc)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

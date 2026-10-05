"""Plot GSR accuracy, time and memory against sequence length.

Four panels, each answering a different question:
  1 GS-HOTA vs length, full window vs common window, attributes on vs off
  2 the decomposition (DetA / AssA / LocA) on the common window -- where degradation lives
  3 wall-clock vs length, against a linear reference -- is the pipeline still linear
  4 peak memory vs length -- extrapolate whether a 45-minute half fits in RAM
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--work", required=True,
                 help="dir holding metrics/sweep_scores.json, metrics/sweep_summary.csv, metrics/mem_*.csv")
_ap.add_argument("--out", default=None, help="output PNG (default <work>/metrics/gsr_length_sweep.png)")
_args = _ap.parse_args()
SP = Path(_args.work)
FPS = 25

scores = json.loads((SP / "metrics/sweep_scores.json").read_text())
if not scores:
    print("no scores yet"); sys.exit(0)

def pick(window, attrs, field):
    rs = [r for r in scores if r["window"] == window and r["attrs"] == attrs]
    rs.sort(key=lambda r: r["frames"])
    return [r["frames"] / FPS / 60 for r in rs], [r[field] for r in rs]

# wall-clock
wall = {}
f = SP / "metrics/sweep_summary.csv"
if f.exists():
    for row in csv.DictReader(open(f)):
        try: wall[row["label"]] = int(row["wall_s"])
        except (ValueError, KeyError): pass

# peak memory per run
peak = {}
for lab in ("30s", "1min", "5min", "15min", "30min"):
    p = SP / f"metrics/mem_{lab}.csv"
    if not p.exists(): continue
    vals, sysu = [], []
    for row in csv.DictReader(open(p)):
        try:
            vals.append(float(row["main_rss_gb"])); sysu.append(float(row["sys_used_gb"]))
        except (ValueError, KeyError): pass
    if vals: peak[lab] = (max(vals), max(sysu))

LAB2MIN = {"30s": 0.5, "1min": 1.0, "5min": 5.0, "15min": 15.0, "30min": 30.0, "45min": 45.08}

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("SoccerTrack v2 GSR: how sequence length affects accuracy, time and memory\n"
             "match 128057, 1st half, nested prefixes of identical footage — 14-module TrackLab pipeline",
             fontsize=12, y=0.98)

# --- 1 GS-HOTA -------------------------------------------------------------
a = ax[0][0]
for window, attrs, style, c, lbl in (
        ("full", "on", "-o", "#c0392b", "full window, attributes ON (official)"),
        ("full", "off", "-s", "#e67e22", "full window, attributes OFF"),
        ("common750", "on", "--o", "#2980b9", "common first 30 s, attributes ON"),
        ("common750", "off", "--s", "#16a085", "common first 30 s, attributes OFF")):
    x, y = pick(window, attrs, "HOTA")
    if x: a.plot(x, y, style, color=c, label=lbl, lw=1.8, ms=6)
a.set_xscale("log"); a.set_xlabel("sequence length (minutes, log)"); a.set_ylabel("GS-HOTA (%)")
a.set_title("1. GS-HOTA vs sequence length"); a.grid(alpha=.3); a.legend(fontsize=8)

# --- 2 decomposition -------------------------------------------------------
a = ax[0][1]
for field, c, m in (("DetA", "#8e44ad", "o"), ("AssA", "#d35400", "s"),
                    ("LocA", "#27ae60", "^"), ("DetRe", "#7f8c8d", "v")):
    x, y = pick("common750", "off", field)
    if x: a.plot(x, y, "-", marker=m, color=c, label=field, lw=1.8, ms=6)
a.set_xscale("log"); a.set_xlabel("sequence length (minutes, log)"); a.set_ylabel("%")
a.set_title("2. Where degradation lives (common 30 s window, attributes off)\n"
            "causal modules cannot degrade; the four global modules can", fontsize=10)
a.grid(alpha=.3); a.legend(fontsize=8)

# --- 3 wall clock ----------------------------------------------------------
a = ax[1][0]
if wall:
    labs = [l for l in ("30s", "1min", "5min", "15min", "30min") if l in wall]
    x = [LAB2MIN[l] for l in labs]; y = [wall[l] / 3600 for l in labs]
    a.plot(x, y, "-o", color="#2c3e50", label="measured", lw=1.8)
    if len(x) >= 1:
        rate = y[0] / x[0]
        xs = np.array([min(x), 45.08])
        a.plot(xs, rate * xs, ":", color="#95a5a6",
               label=f"linear from shortest ({rate:.2f} h per video-minute)")
        a.annotate(f"45-min extrapolation: {rate*45.08:.1f} h",
                   xy=(45.08, rate * 45.08), fontsize=8, ha="right", va="bottom")
a.set_xscale("log"); a.set_xlabel("sequence length (minutes, log)")
a.set_ylabel("wall clock (hours)")
a.set_title("3. Runtime vs length (quadratic merge already patched)"); a.grid(alpha=.3)
a.legend(fontsize=8)

# --- 4 memory --------------------------------------------------------------
a = ax[1][1]
if peak:
    labs = [l for l in ("30s", "1min", "5min", "15min", "30min") if l in peak]
    x = [LAB2MIN[l] for l in labs]
    a.plot(x, [peak[l][0] for l in labs], "-o", color="#c0392b", label="main process RSS", lw=1.8)
    a.plot(x, [peak[l][1] for l in labs], "-s", color="#2980b9", label="system used", lw=1.8)
    a.axhline(61, ls="--", color="k", lw=1, label="physical RAM (61 GB)")
    if len(x) >= 2:
        sl = np.polyfit(x, [peak[l][0] for l in labs], 1)
        a.plot([min(x), 45.08], np.polyval(sl, [min(x), 45.08]), ":", color="#95a5a6",
               label=f"linear fit -> {np.polyval(sl, 45.08):.1f} GB at 45 min")
a.set_xscale("log"); a.set_xlabel("sequence length (minutes, log)"); a.set_ylabel("GB")
a.set_title("4. Peak memory vs length"); a.grid(alpha=.3); a.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(_args.out) if _args.out else SP / "metrics/gsr_length_sweep.png"
plt.savefig(out, dpi=150)
print(f"wrote {out}")

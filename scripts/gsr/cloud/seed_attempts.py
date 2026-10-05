#!/usr/bin/env python3
"""Seed ckpts/.attempts from half.log when the runner that wrote the checkpoints predates the
attempts log (old in-memory code). Each '=== pipeline attempt N/M (...) <ISO>Z ===' line opens an
attempt; the module list is the pipeline=[...] of the following 'resuming with:' line, else the
full pipeline. Writes '<epoch> <modules...>' lines. Idempotent: no-op if .attempts is non-empty."""
import re, sys, os, datetime
FULL = "bbox_detector pose_detector reid track jersey_number_det jersey_number_rec calibration gta main_subject_filter gk_role_assignment consistency tracklet_agg team team_side"
log, out = sys.argv[1], sys.argv[2]
if os.path.exists(out) and os.path.getsize(out) > 0:
    print("attempts log already present; nothing to seed"); sys.exit(0)
txt = open(log, errors="replace").read().replace("\r", "\n")
lines = txt.split("\n")
rows = []
for i, ln in enumerate(lines):
    m = re.search(r"=== pipeline attempt \d+/\d+ .*?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z ===", ln)
    if not m: continue
    epoch = int(datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp())
    mods = FULL
    for nxt in lines[i+1:i+3]:
        pm = re.search(r"pipeline=\[([^\]]*)\]", nxt)
        if pm: mods = pm.group(1).replace(",", " "); break
    rows.append(f"{epoch} {mods}")
if not rows:
    print("no attempt lines found; nothing to seed"); sys.exit(0)
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write("\n".join(rows) + "\n")
print(f"seeded {len(rows)} attempt(s) into {out}")

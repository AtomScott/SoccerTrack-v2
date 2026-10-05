#!/usr/bin/env python3
"""Convert cloud runners' score.log files into the canonical score-JSON schema.

The fleet instances score with score_one.py but persist a human-readable
score.log plus a bare `gshota` number. This walks every pulled result dir
(gsr-cloud-results/CLPD-* and gsr2/CLPD-*), parses the two metric blocks, and
writes `zscore.json` (z so pred.json sorts first alphabetically but is skipped
by size) next to each log so compile_full_table.py picks the half up.
Idempotent; skips dirs without a parseable score.log.
"""

import json
import re
from pathlib import Path

CLOUD_ROOT = Path("/mnt/storage/SoccerTrack-v2/gsr-cloud-results")

BLOCK = re.compile(
    r"attributes (ON \(official GS-HOTA\)|OFF \(geometry \+ association\))\s*\n"
    r"\s*GS-HOTA\s+([\d.]+)\s+DetA\s+([\d.]+)\s+AssA\s+([\d.]+)\s+LocA\s+([\d.]+)")
DROPPED = re.compile(r"dropped ([\d,]+) \(")
NPRED = re.compile(r"([\d,]+) object predictions")


def convert(d):
    log = d / "score.log"
    out = d / "zscore.json"
    if not log.exists() or out.exists():
        return False
    text = log.read_text(errors="replace")
    blocks = BLOCK.findall(text)
    if len(blocks) != 2:
        return False
    scores = {}
    for cond, hota, deta, assa, loca in blocks:
        key = ("attributes ON (official GS-HOTA)" if cond.startswith("ON")
               else "attributes OFF (geometry + association)")
        scores[key] = {"HOTA": float(hota), "DetA": float(deta),
                       "AssA": float(assa), "LocA": float(loca)}
    payload = {"seq": d.name, "scores": scores, "source": "score.log"}
    m = DROPPED.search(text)
    if m:
        payload["dropped_non_finite"] = int(m.group(1).replace(",", ""))
    m = NPRED.search(text)
    if m:
        payload["n_object_predictions"] = int(m.group(1).replace(",", ""))
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    return True


def main():
    n = 0
    for d in sorted(CLOUD_ROOT.glob("CLPD-*")) + sorted(
            (CLOUD_ROOT / "gsr2").glob("CLPD-*")) + sorted(
            (CLOUD_ROOT / "gsr4").glob("CLPD-*")):
        if d.is_dir():
            n += convert(d)
    print(f"converted {n} new dir(s)")


if __name__ == "__main__":
    main()

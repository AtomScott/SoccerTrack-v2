#!/usr/bin/env python3
"""Compile the prefix rescoring of every whole-half run into one CSV and print
the corpus means per prefix.

Input: results/gsr/prefix_rescoring/CLPD-<match>-<half>.json written by
score_prefixes.py (the whole-half predictions of Table 4 scored over their
first N frames, N in 750, 1500, 3000, 7500, 15000, 30000 and the full half).
Output: results/gsr/prefix_rescoring_all_halves.csv, one row per half and
prefix, and a console summary: mean over the twenty halves per prefix
(official and attributes off), the independent 30 s runs' mean for
comparison, and a check that each half's full-length rescoring reproduces
its Table 4 row.
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "results" / "gsr" / "prefix_rescoring"
OUT_CSV = REPO / "results" / "gsr" / "prefix_rescoring_all_halves.csv"
FULL_CSV = REPO / "results" / "gsr" / "full_table_all_matches.csv"
SWEEP_CSV = REPO / "results" / "gsr" / "sweep30s_all_matches.csv"
ON = "attributes ON (official GS-HOTA)"
OFF = "attributes OFF (geometry + association)"
LABELS = {750: "30s", 1500: "1min", 3000: "2min", 7500: "5min", 15000: "10min", 30000: "20min"}


def main():
    rows = []
    for p in sorted(IN_DIR.glob("CLPD-*.json")):
        j = json.load(open(p))
        seq = j["seq"]; _, match, half = seq.split("-")
        total = j["n_frames_total"]
        for n, r in sorted(((int(k), v) for k, v in j["prefixes"].items())):
            on, off = r[ON], r[OFF]
            rows.append({"seq": seq, "match": match, "half": half, "n_frames": n,
                         "label": "full" if n == total else LABELS.get(n, str(n)),
                         "hota": on["HOTA"], "deta": on["DetA"], "assa": on["AssA"], "loca": on["LocA"],
                         "off_hota": off["HOTA"], "off_deta": off["DetA"], "off_assa": off["AssA"], "off_loca": off["LocA"],
                         "ids": on["ids"], "gt_ids": on["gt_ids"]})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    halves = sorted({r["seq"] for r in rows})
    print(f"wrote {OUT_CSV}  ({len(halves)} halves, {len(rows)} rows)")

    full = {(r["match"], r["half"]): r for r in csv.DictReader(open(FULL_CSV)) if r["chosen"] == "True"}
    sweep = {(r["match"], r["half"]): r for r in csv.DictReader(open(SWEEP_CSV))}
    bad = []
    for r in rows:
        if r["label"] != "full":
            continue
        t = full[(r["match"], r["half"])]
        if abs(r["hota"] - float(t["hota"])) > 0.006 or abs(r["off_hota"] - float(t["off_hota"])) > 0.006:
            bad.append((r["seq"], r["hota"], t["hota"], r["off_hota"], t["off_hota"]))
    print("full-length rescoring reproduces Table 4:", "yes" if not bad else f"NO {bad}")

    print(f"\n{'prefix':>8} {'n':>3} {'official':>9} {'min':>6} {'max':>6}   {'attrs off':>9} {'min':>6} {'max':>6}")
    for label in ["30s", "1min", "2min", "5min", "10min", "20min", "full"]:
        sel = [r for r in rows if r["label"] == label]
        if not sel:
            continue
        on = [r["hota"] for r in sel]; off = [r["off_hota"] for r in sel]
        print(f"{label:>8} {len(sel):>3} {sum(on)/len(on):9.2f} {min(on):6.2f} {max(on):6.2f}   {sum(off)/len(off):9.2f} {min(off):6.2f} {max(off):6.2f}")
    s_on = [float(sweep[k]["gshota_official"]) for k in sweep]; s_off = [float(sweep[k]["gshota_attrs_off"]) for k in sweep]
    print(f"{'indep30s':>8} {len(s_on):>3} {sum(s_on)/len(s_on):9.2f} {min(s_on):6.2f} {max(s_on):6.2f}   {sum(s_off)/len(s_off):9.2f} {min(s_off):6.2f} {max(s_off):6.2f}")
    # monotonicity per half (official)
    non_mono = []
    for seq in halves:
        seqrows = sorted((r for r in rows if r["seq"] == seq), key=lambda r: r["n_frames"])
        h = [r["hota"] for r in seqrows]
        if any(b > a + 1e-9 for a, b in zip(h, h[1:])):
            non_mono.append(seq)
    print(f"\nhalves whose official score rises somewhere along the prefixes: {len(non_mono)} of {len(halves)}")
    print("every half below its 750-frame rescoring at full length:", all(
        [r for r in rows if r["seq"] == s and r["label"] == "full"][0]["hota"] <
        [r for r in rows if r["seq"] == s and r["label"] == "30s"][0]["hota"] for s in halves))


if __name__ == "__main__":
    main()

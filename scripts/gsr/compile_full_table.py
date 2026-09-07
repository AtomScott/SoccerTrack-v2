#!/usr/bin/env python3
"""Compile the full 20-half GS-HOTA table from cloud + local score JSONs.

Sources, in the order a half is looked up:
  fleet-1: results/gsr/cloud/fleet1/CLPD-<M>-<H>/*.json, falling back to
           /mnt/storage/SoccerTrack-v2/gsr-cloud-results/CLPD-<M>-<H>/*.json
  fleet-2: results/gsr/cloud/fleet2/CLPD-<M>-<H>/*.json, falling back to
           /mnt/storage/SoccerTrack-v2/gsr-cloud-results/gsr2/CLPD-<M>-<H>/*.json
  local:   results/gsr/score_45min_<M>_<H>.json   (128057 was run locally;
           132831 local files are the pred-shift rescoring, used only as a
           fallback and flagged as such)

The results/gsr/cloud tree holds versioned copies of each completed cloud
half's small artefacts (zscore.json, summary.txt, score.log, staging.log,
half.log, gshota; never pred.json) plus the two fleet STATUS manifests, so
that the paper tables regenerate from a clean checkout without the storage
mount. When a further cloud half completes, copy its artefacts there.

When both fleets have a half, the first finisher is reported and the
fleet1-vs-fleet2 GS-HOTA delta is recorded in the CSV (fp16 was validated
score-identical on the 30 s smoke; this is the production check).

Outputs (default under results/gsr/):
  full_table_all_matches.csv   one row per half per source, plus chosen flag
  full_table_rows.tex          LaTeX body rows in the tab:gsr_results format
Prints a human-readable table with per-match means to stdout.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
HALVES = ["1st", "2nd"]

CLOUD_ROOT = Path("/mnt/storage/SoccerTrack-v2/gsr-cloud-results")
LOCAL_DIR = Path(__file__).resolve().parents[2] / "results" / "gsr"
CLOUD_LOCAL = LOCAL_DIR / "cloud"
# source name -> (in-repo subdir, subdir under the storage mount)
FLEETS = {"fleet1": ("fleet1", ""), "fleet2": ("fleet2", "gsr2")}

ON_KEY = "attributes ON (official GS-HOTA)"
OFF_KEY = "attributes OFF (geometry + association)"


def load_score(path):
    try:
        d = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if "scores" not in d or ON_KEY not in d.get("scores", {}):
        return None
    on, off = d["scores"][ON_KEY], d["scores"][OFF_KEY]
    return {
        "hota": on["HOTA"], "deta": on["DetA"], "assa": on["AssA"],
        "loca": on["LocA"], "off_hota": off["HOTA"], "off_deta": off["DetA"],
        "off_assa": off["AssA"], "off_loca": off["LocA"],
        "dropped": d.get("dropped_non_finite"), "path": str(path),
    }


def find_in_dir(dirpath):
    if not dirpath.is_dir():
        return None
    for p in sorted(dirpath.glob("*.json")):
        # Cloud dirs also hold pred.json (hundreds of MB); never parse those.
        if p.name == "pred.json" or p.stat().st_size > 5_000_000:
            continue
        s = load_score(p)
        if s:
            return s
    return None


def collect(match, half):
    """Return {source_name: score_dict} for every source that has this half."""
    seq = f"CLPD-{match}-{half}"
    out = {}
    for name, (repo_sub, mount_sub) in FLEETS.items():
        s = find_in_dir(CLOUD_LOCAL / repo_sub / seq) or find_in_dir(CLOUD_ROOT / mount_sub / seq)
        if s:
            out[name] = s
    p = LOCAL_DIR / f"score_45min_{match}_{half}.json"
    if p.exists():
        s = load_score(p)
        if s:
            out["local"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=LOCAL_DIR)
    ap.add_argument("--require-complete", action="store_true",
                    help="exit 1 if any of the 20 halves has no score yet")
    args = ap.parse_args()

    rows, chosen_rows, missing = [], [], []
    for match in MATCHES:
        for half in HALVES:
            sources = collect(match, half)
            if not sources:
                missing.append(f"{match}-{half}")
                continue
            # Cloud reruns supersede the local 132831 pred-shift rescoring;
            # 128057 exists only locally, so "local" is authoritative there.
            for pref in ("fleet1", "fleet2", "local"):
                if pref in sources:
                    chosen = pref
                    break
            delta = ""
            if "fleet1" in sources and "fleet2" in sources:
                delta = f"{sources['fleet1']['hota'] - sources['fleet2']['hota']:+.3f}"
            for name, s in sources.items():
                rows.append({"match": match, "half": half, "source": name,
                             "chosen": name == chosen,
                             "fleet_delta_hota": delta if name == chosen else "",
                             **{k: v for k, v in s.items() if k != "path"},
                             "path": s["path"]})
            chosen_rows.append({"match": match, "half": half,
                                "source": chosen, **sources[chosen]})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "full_table_all_matches.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["match", "half", "source"])
        w.writeheader()
        w.writerows(rows)

    # LaTeX body in the tab:gsr_results column order:
    # Half & GS-HOTA & DetA & AssA & LocA & Attrs off
    metrics = ["hota", "deta", "assa", "loca", "off_hota"]
    tex_lines = []
    for match in MATCHES:
        halves = [r for r in chosen_rows if r["match"] == match]
        for r in halves:
            tex_lines.append(
                f"    {r['match']}, {r['half']} & "
                + " & ".join(f"{r[m]:.2f}" for m in metrics) + r" \\")
    if chosen_rows:
        tex_lines.append(r"    \midrule")
        tex_lines.append(
            "    Mean & " + " & ".join(
                f"{sum(r[m] for r in chosen_rows) / len(chosen_rows):.2f}"
                for m in metrics) + r" \\")
    tex_path = args.out_dir / "full_table_rows.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")

    # Human-readable summary with per-match means.
    print(f"{'half':<14} {'src':<7} {'GS-HOTA':>8} {'DetA':>7} {'AssA':>7} "
          f"{'LocA':>7} {'off':>7}")
    for match in MATCHES:
        halves = [r for r in chosen_rows if r["match"] == match]
        if not halves:
            continue
        for r in halves:
            print(f"{r['match']}-{r['half']:<7} {r['source']:<7} "
                  f"{r['hota']:>8.2f} {r['deta']:>7.2f} {r['assa']:>7.2f} "
                  f"{r['loca']:>7.2f} {r['off_hota']:>7.2f}")
        if len(halves) == 2:
            print(f"{match} mean{'':<3} {'':<7} "
                  + " ".join(f"{sum(r[m] for r in halves) / 2:>{w}.2f}"
                             for m, w in zip(metrics, (8, 7, 7, 7, 7))))
    if chosen_rows:
        n = len(chosen_rows)
        print(f"\noverall mean ({n} halves): "
              + " ".join(f"{m}={sum(r[m] for r in chosen_rows) / n:.2f}"
                         for m in metrics))
    both = [r for r in rows if r["fleet_delta_hota"]]
    if both:
        print("fleet1-vs-fleet2 GS-HOTA deltas: "
              + ", ".join(f"{r['match']}-{r['half']}: {r['fleet_delta_hota']}"
                          for r in both))
    if missing:
        print(f"\nMISSING ({len(missing)}/20): {', '.join(missing)}")
        if args.require_complete:
            sys.exit(1)
    print(f"\nwrote {csv_path}\nwrote {tex_path}")


if __name__ == "__main__":
    main()

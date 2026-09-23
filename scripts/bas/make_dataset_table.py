"""Emit the dataset and split table for the BAS benchmark.

Answers items (a)-(d) of the TODO in paper/sections/02_results.tex from the released files
rather than from the release notes: per-match duration and frame count, the split, the
benchmark event count, actor-link coverage, and the two per-match data limitations that
affect how the results must be read.

WHAT THIS TABLE HAS TO SAY THAT A PLAIN INVENTORY WOULD NOT

  * Periods played is 3 for three matches, and only two of them were filmed. The events in
    the unfilmed period are annotated but outside the benchmark, so an events column that
    quoted the annotation count would not match the number actually scored.
  * The ball track is clamped to the pitch rectangle in 132831 and 132877, which changes what
    a with-ball result means on those matches. 132831 is in the test split.
  * The panoramic videos are not a single resolution.

USAGE
    python scripts/bas/make_dataset_table.py --out results/bas
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_utils.bas_periods import period_and_frame  # noqa: E402

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
TEST = {"128057", "132831"}
VAL = {"117093", "132877"}
HALF_NAME = {1: "1st", 2: "2nd"}
BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
_CANON = {l.casefold(): l for l in BAS_LABELS}


def split_of(m: str) -> str:
    return "test" if m in TEST else ("val" if m in VAL else "train")


def video_res(data: Path, m: str) -> str:
    out = set()
    for h in ("1st", "2nd"):
        p = data / "interim" / m / f"{m}_panorama_{h}_half.mp4"
        if not p.exists():
            continue
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True).stdout.strip()
        if r:
            out.add(r.replace(",", r"$\times$"))
    return " / ".join(sorted(out)) if out else "--"


def ball_clamped(ball_dir: Path, m: str) -> bool:
    """True when the ball never leaves the pitch, which no real ball track does."""
    off = 0
    tot = 0
    for h in ("1st", "2nd"):
        p = ball_dir / f"{m}_{h}_ball.npz"
        if not p.exists():
            return False
        d = np.load(p)
        x, y = d["x"], d["y"]
        ok = ~np.isnan(x)
        off += int(((np.abs(x[ok]) > 52.5) | (np.abs(y[ok]) > 34.0)).sum())
        tot += int(ok.sum())
    return tot > 0 and off == 0


def collect(data: Path, ball_dir: Path, table: dict) -> list[dict]:
    rows = []
    for m in MATCHES:
        tm = table["matches"][m]
        meta = (data / "production" / "raw" / m /
                f"{m}_tracker_box_metadata.xml").read_text()
        title = re.search(r'matchTitleEn="([^"]*)"', meta)
        if title is None:
            title = re.search(r'matchTitle="([^"]*)"', meta)
        date = re.search(r'matchDatetime="([0-9]{4}-[0-9]{2}-[0-9]{2})', meta)
        acts = json.loads((data / "production" / "bas" / m /
                           f"{m}_12_class_events.json").read_text())
        acts = acts.get("actions", acts.get("annotations"))
        per = Counter()
        linked = bench = 0
        for a in acts:
            p, _ = period_and_frame(a["gameTime"], int(a["position"]), tm["periods"])
            per[p] += 1
            if p in (1, 2):
                bench += 1
                if a.get("player_id") not in (None, ""):
                    linked += 1
        n_frames = sum(tm["periods"][str(h)]["n_frames"] or 0 for h in (1, 2))
        rows.append({
            "match": m, "split": split_of(m),
            "title": (title.group(1) if title else ""),
            "date": (date.group(1) if date else ""),
            "periods_played": sum(1 for p, n in per.items() if n >= 50),
            "frames": n_frames, "minutes": n_frames / 25 / 60,
            "annotated": len(acts), "benchmark": bench,
            "unfilmed": len(acts) - bench,
            "link_pct": 100.0 * linked / max(bench, 1),
            "res": video_res(data, m),
            "ball_clamped": ball_clamped(ball_dir, m),
        })
    return rows


def latex(rows: list[dict]) -> str:
    tot = {k: sum(r[k] for r in rows) for k in ("frames", "annotated", "benchmark", "unfilmed")}
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{The ten SoccerTrack v2 matches, their split assignment and the BAS "
         r"annotation counts. \emph{Periods} is the number of $45\,\text{min}$ periods "
         r"played; three matches were played as three periods and only the first two were "
         r"filmed and tracked, so the \emph{Benchmark} column counts the events with a "
         r"corresponding video and game-state annotation and \emph{Unfilmed} counts those "
         r"without. \emph{Link} is the fraction of benchmark events naming their actor. "
         r"$\ast$ marks the two matches whose recorded ball track is clamped to the pitch "
         r"rectangle and therefore never leaves play.}",
         r"  \label{tab:bas_dataset}", r"  \small",
         r"  \begin{tabular}{llrrrrrrl}", r"    \toprule",
         r"    Match & Split & Periods & Frames & Min. & Annot. & Benchmark & Unfilmed & "
         r"Link \\", r"    \midrule"]
    for r in rows:
        star = r"$^\ast$" if r["ball_clamped"] else ""
        L.append(f'    {r["match"]}{star} & {r["split"]} & {r["periods_played"]} & '
                 f'{r["frames"]:,} & {r["minutes"]:.0f} & {r["annotated"]:,} & '
                 f'{r["benchmark"]:,} & {r["unfilmed"]:,} & {r["link_pct"]:.1f}\\% \\\\')
    L.append(r"    \midrule")
    link = 100.0 * sum(r["link_pct"] * r["benchmark"] for r in rows) / tot["benchmark"] / 100
    L.append(f'    Total & & & {tot["frames"]:,} & {tot["frames"]/25/60:.0f} & '
             f'{tot["annotated"]:,} & {tot["benchmark"]:,} & {tot["unfilmed"]:,} & '
             f'{link:.1f}\\% \\\\')
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--ball", default="/data/share/SoccerTrack-v2/data/derived/bas/ball")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--out", default="results/bas")
    a = ap.parse_args()
    table = json.loads(Path(a.periods).read_text())
    rows = collect(Path(a.data), Path(a.ball), table)

    hdr = (f'{"match":8} {"split":6} {"per":>3} {"frames":>9} {"min":>4} {"annot":>7} '
           f'{"bench":>7} {"unfilmed":>8} {"link":>6} {"ball":>8}  {"resolution":<22} title')
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f'{r["match"]:8} {r["split"]:6} {r["periods_played"]:3} {r["frames"]:9,} '
              f'{r["minutes"]:4.0f} {r["annotated"]:7,} {r["benchmark"]:7,} '
              f'{r["unfilmed"]:8,} {r["link_pct"]:5.1f}% '
              f'{"CLAMPED" if r["ball_clamped"] else "ok":>8}  '
              f'{r["res"].replace(chr(92)+"times","x").replace("$",""):<22} {r["title"]}')
    tot = {k: sum(r[k] for r in rows) for k in ("frames", "annotated", "benchmark", "unfilmed")}
    print("-" * len(hdr))
    print(f'{"TOTAL":8} {"":6} {"":3} {tot["frames"]:9,} {tot["frames"]/25/60:4.0f} '
          f'{tot["annotated"]:7,} {tot["benchmark"]:7,} {tot["unfilmed"]:8,}')
    print(f'\nsplit sizes (benchmark events): ' + ", ".join(
        f'{s} {sum(r["benchmark"] for r in rows if r["split"] == s):,} '
        f'({sum(1 for r in rows if r["split"] == s)} matches)'
        for s in ("train", "val", "test")))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tab_bas_dataset.tex").write_text(latex(rows) + "\n")
    print(f"\nwrote {out}/tab_bas_dataset.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())

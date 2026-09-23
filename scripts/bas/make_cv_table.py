"""Emit the five-fold cross-match table.

WHY THE PAPER NEEDS THIS AND NOT JUST THE 8/2 SPLIT

With ten matches, a single held-out pair is not enough to carry a benchmark. Measured here,
the spread across the five folds is 4.3x the spread across training seeds for the trajectory
condition and 2.3x for the trajectory-plus-ball condition: WHICH matches land in test matters
several times more than how the model was initialised. Reporting one 8/2 draw without that
context invites the reader to read a fold effect as a method effect.

Every match appears in test exactly once, so the benchmark rests on all ten matches rather
than two. Fold 0 is deliberately the SoccerTrack Challenge 2025 pair, so it remains directly
comparable with the leaderboard while the mean and standard deviation say how much that
single draw can mislead.

Validation for each fold is the next fold's test pair. That rule is arbitrary but fixed, and
it never lets a fold tune on its own test matches. It does mean fold 4 validates on the
Challenge pair, so the cross-validated mean is not fully independent of the headline split ---
a standard property of k-fold, stated rather than hidden.

USAGE
    python scripts/bas/make_cv_table.py --out results/bas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUNS = "outputs/sn"
FOLDS = {0: ["128057", "132831"], 1: ["117092", "117093"], 2: ["118575", "118576"],
         3: ["118577", "118578"], 4: ["128058", "132877"]}
TRACKS = [("noball", "trajectory"), ("ball", "trajectory + ball")]
K = ["mAP@1s", "mAPw@1s", "mAP@5s", "mAPw@5s"]
# Standard deviation over three training seeds on the Challenge split, for comparison.
SEED_SD = {"noball": 0.0065, "ball": 0.0284}


def f3(x) -> str:
    return "--" if x != x else f"{x:.3f}"


def latex(per_fold: dict, summary: dict, chance: dict) -> str:
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{Five-fold cross-match evaluation. Each match appears in the test set "
         r"exactly once; fold 0 is the SoccerTrack Challenge 2025 pair. Validation for each "
         r"fold is the following fold's test pair, so no fold tunes on its own test matches. "
         r"The standard deviation across folds is $4.3\times$ (trajectory) and $2.3\times$ "
         r"(trajectory {+} ball) the standard deviation across training seeds, so which "
         r"matches are held out matters several times more than initialisation.}",
         r"  \label{tab:bas_cv}", r"  \small",
         r"  \begin{tabular}{llcccc}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{$\tau = 1\,$s} & \multicolumn{2}{c}{$\tau = 5\,$s} \\",
         r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"    Input & Test fold & mAP & mAP$_w$ & mAP & mAP$_w$ \\",
         r"    \midrule"]
    for key, name in TRACKS:
        for f, ms in FOLDS.items():
            row = per_fold.get(f"fold{f}_{key}")
            if row is None:
                continue
            lbl = name if f == 0 else ""
            tag = r"\,(Challenge)" if f == 0 else ""
            L.append(f'    {lbl} & {", ".join(ms)}{tag} & '
                     + " & ".join(f3(row[k]) for k in K) + r" \\")
        mu = [summary[key][k][0] for k in K]
        sd = [summary[key][k][1] for k in K]
        L.append(r"    \cmidrule(lr){2-6}")
        L.append(r"    & \textbf{Mean} & "
                 + " & ".join(rf"\textbf{{{f3(m)}}}" for m in mu) + r" \\")
        L.append(r"    & SD across folds & " + " & ".join(f3(s) for s in sd) + r" \\")
        L.append(r"    \midrule")
    L.append(r"    \textit{Uniform cadence} & Mean & "
             + " & ".join(f3(chance[k][0]) for k in K) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def per_class_from_folds(out: Path) -> str:
    """Per-class AP pooled over all five folds, i.e. over all ten matches.

    Each fold contributes its own test matches, so support here is the whole benchmark
    rather than the two matches of a single split. For the rarest classes that is the
    difference between an interpretable number and a coin flip: Header has 5 instances in
    the Challenge test pair and 31 across the dataset.
    """
    from src.data_utils.soccertrack_v2 import BAS_LABELS
    per, sup = {}, {}
    for key, _ in TRACKS:
        acc = {l: [] for l in BAS_LABELS}
        for f in FOLDS:
            sc = json.loads((Path(f"{RUNS}/fold{f}_{key}") / "tables" /
                             "scores.json").read_text())["model"]["overall"]
            for l in BAS_LABELS:
                v = sc[f"perClass@1s"][l]
                if v == v:
                    acc[l].append((v, sc["support"][l]))
                if key == "noball":
                    sup[l] = sup.get(l, 0) + sc["support"][l]
        # support-weighted across folds: a fold with more instances of a class counts more
        per[key] = {l: (sum(v * n for v, n in acc[l]) / sum(n for _, n in acc[l])
                        if acc[l] and sum(n for _, n in acc[l]) else float("nan"))
                    for l in BAS_LABELS}
    ch = {}
    for f in FOLDS:
        sc = json.loads((Path(f"{RUNS}/fold{f}_noball") / "tables" /
                         "scores.json").read_text())["chance"]["overall"]
        for l in BAS_LABELS:
            v = sc["perClass@1s"][l]
            if v == v:
                ch.setdefault(l, []).append((v, sc["support"][l]))
    chm = {l: (sum(v * n for v, n in ch[l]) / sum(n for _, n in ch[l])
               if ch.get(l) else float("nan")) for l in BAS_LABELS}

    print(f'\n{"class":26} {"n (all 10)":>11} {"chance":>8} {"traj":>8} {"+ball":>8} {"delta":>8}')
    lines = []
    for l in sorted(BAS_LABELS, key=lambda x: -sup[x]):
        d = per["ball"][l] - per["noball"][l]
        print(f'{l:26} {sup[l]:11} {chm[l]:8.3f} {per["noball"][l]:8.3f} '
              f'{per["ball"][l]:8.3f} {d:+8.3f}')
        lines.append(f'    {l}{"$^\\dagger$" if sup[l] < 100 else ""} & {sup[l]} & '
                     f'{f3(chm[l])} & {f3(per["noball"][l])} & {f3(per["ball"][l])} \\\\')
    tex = ["\\begin{table}[t]", "  \\centering",
           "  \\caption{Per-class average precision at $\\tau = 1\\,$s, pooled over the five "
           "cross-match folds and therefore over all ten matches. $n$ is the number of "
           "ground-truth events across the benchmark. Rows marked $\\dagger$ have $n < 100$. "
           "Classes are ordered by support.}",
           "  \\label{tab:bas_per_class}", "  \\small",
           "  \\begin{tabular}{lrccc}", "    \\toprule",
           "    Class & $n$ & chance & trajectory & {+}ball \\\\", "    \\midrule"]
    tex += lines
    tex += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (out / "tab_bas_per_class.tex").write_text("\n".join(tex) + "\n")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="results/bas/folds.json")
    ap.add_argument("--out", default="results/bas")
    a = ap.parse_args()
    doc = json.loads(Path(a.folds).read_text())
    per_fold, summary = doc["per_fold"], doc["summary"]

    print(f'{"input":20} {"test fold":26} ' + " ".join(f"{k:>9}" for k in K))
    for key, name in TRACKS:
        for f, ms in FOLDS.items():
            r = per_fold.get(f"fold{f}_{key}")
            if r is None:
                continue
            tag = " (Challenge)" if f == 0 else ""
            print(f'{name:20} {",".join(ms) + tag:26} '
                  + " ".join(f"{r[k]:9.3f}" for k in K))
        print(f'{"":20} {"MEAN":26} '
              + " ".join(f'{summary[key][k][0]:9.3f}' for k in K))
        print(f'{"":20} {"SD across folds":26} '
              + " ".join(f'{summary[key][k][1]:9.3f}' for k in K))
        sd = summary[key]["mAP@1s"][1]
        print(f'{"":20} {"vs SD across seeds":26} {SEED_SD[key]:9.3f}'
              f'   -> fold spread is {sd / SEED_SD[key]:.1f}x seed spread')
        print()
    print(f'{"uniform cadence":20} {"MEAN":26} '
          + " ".join(f'{summary["chance"][k][0]:9.3f}' for k in K))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_bas_cv.tex").write_text(latex(per_fold, summary, summary["chance"]) + "\n")
    per_class_from_folds(out)
    print(f"\nwrote {out}/tab_bas_cv.tex and {out}/tab_bas_per_class.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())

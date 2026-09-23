"""Emit the paper's two BAS tables, both input conditions, averaged over seeds.

Supersedes the single-condition output of make_tables.py. Two differences that matter:

  * BOTH tracks in one table. The paper reports a trajectory-only condition (reproducible
    from the released game state alone) and a trajectory-plus-ball condition, so the
    comparison belongs in one table rather than two.
  * PER-CLASS FIGURES ARE SEED MEANS, like the aggregates. Quoting aggregates as a mean over
    seeds and per-class rows from a single run invites the reader to combine numbers that
    were not computed the same way.

The per-match decomposition is not optional presentation. The ball is intact in 128057 and
clamped to the pitch rectangle in 132831, so the pooled with-ball figure averages two
incompatible regimes; the table carries both rows so the pooled number cannot be quoted
alone.

USAGE
    python scripts/bas/make_final_tables.py --runs outputs/final --out results/bas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_utils.soccertrack_v2 import BAS_LABELS  # noqa: E402

TEST = ["128057", "132831"]
TOLS = [1, 5]
LOW_SUPPORT = 30
TRACKS = [("noball", "trajectory"), ("ball", "trajectory + ball")]


def load_seeds(runs: Path, prefix: str, seeds=(0, 1, 2)) -> list[dict]:
    out = []
    for s in seeds:
        p = runs / f"{prefix}_seed{s}" / "tables" / "scores.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    if not out:
        raise FileNotFoundError(f"no scored seeds under {runs}/{prefix}_seed*")
    return out


def agg(docs: list[dict], where: str, key: str) -> tuple[float, float, float]:
    v = np.array([d["model"][where][key] for d in docs], float)
    return float(v.mean()), float(v.min()), float(v.max())


def agg_class(docs: list[dict], tol: int, label: str) -> float:
    v = np.array([d["model"]["overall"][f"perClass@{tol}s"][label] for d in docs], float)
    return float(np.nanmean(v))


def f3(x: float) -> str:
    return "--" if x != x else f"{x:.3f}"


def results_table(byt: dict, chance: dict, n_seeds: int) -> str:
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{Ball action spotting on the SoccerTrack v2 test split, from "
         r"ground-truth player trajectories. $\text{mAP}$ is the unweighted mean over the "
         r"twelve classes and $\text{mAP}_w$ weights each class by its support. Figures are "
         rf"means over {n_seeds} training seeds. The uniform-cadence row emits spots at each "
         r"class's mean training-set rate and knows nothing else about the match. The ball "
         r"track is intact in 128057 and clamped to the pitch rectangle in 132831, so the "
         r"pooled ball figure averages two different regimes.}",
         r"  \label{tab:bas_results}", r"  \small",
         r"  \begin{tabular}{llcccc}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{$\tau = 1\,$s} & \multicolumn{2}{c}{$\tau = 5\,$s} \\",
         r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"    Input & Match & mAP & mAP$_w$ & mAP & mAP$_w$ \\",
         r"    \midrule"]
    for key, name in TRACKS:
        docs = byt[key]
        for i, m in enumerate(TEST + ["overall"]):
            label = name if i == 0 else ""
            shown = "Pooled" if m == "overall" else m
            cells = []
            for tol in TOLS:
                for k in (f"mAP@{tol}s", f"mAPw@{tol}s"):
                    mean, _, _ = agg(docs, m, k)
                    cells.append(rf"\textbf{{{f3(mean)}}}" if m == "overall" else f3(mean))
            L.append(f"    {label} & {shown} & " + " & ".join(cells) + r" \\")
        L.append(r"    \midrule")
    ch = chance["chance"]["overall"]
    L.append(r"    \textit{Uniform cadence} & Pooled & "
             + " & ".join(f3(ch[f"mAP{w}@{t}s"]) for t in TOLS for w in ("", "w"))
             + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def per_class_table(byt: dict, chance: dict, n_seeds: int) -> str:
    o = byt["noball"][0]["model"]["overall"]
    ch = chance["chance"]["overall"]
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{Per-class average precision on the SoccerTrack v2 test split, means "
         rf"over {n_seeds} seeds. $n$ is the number of ground-truth events. Rows marked "
         r"$\dagger$ have $n < 30$, where average precision takes only a few distinct values "
         r"and should not be read as a measurement. Classes are ordered by support.}",
         r"  \label{tab:bas_per_class}", r"  \small",
         r"  \begin{tabular}{lrcccccc}", r"    \toprule",
         r"    & & \multicolumn{3}{c}{AP@$1\,$s} & \multicolumn{3}{c}{AP@$5\,$s} \\",
         r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"    Class & $n$ & chance & traj. & {+}ball & chance & traj. & {+}ball \\",
         r"    \midrule"]
    for lab in sorted(BAS_LABELS, key=lambda l: -o["support"][l]):
        n = o["support"][lab]
        mark = r"$^\dagger$" if n < LOW_SUPPORT else ""
        cells = []
        for tol in TOLS:
            cells.append(f3(ch[f"perClass@{tol}s"][lab]))
            for key, _ in TRACKS:
                cells.append(f3(agg_class(byt[key], tol, lab)))
        L.append(f"    {lab}{mark} & {n} & " + " & ".join(cells) + r" \\")
    L.append(r"    \midrule")
    for name, k in (("Mean (12 classes)", "mAP"), ("Support-weighted mean", "mAPw")):
        cells = []
        for tol in TOLS:
            cells.append(f3(ch[f"{k}@{tol}s"]))
            for key, _ in TRACKS:
                cells.append(f3(agg(byt[key], "overall", f"{k}@{tol}s")[0]))
        L.append(f"    {name} & {o['n_gt']:,} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="outputs/final")
    ap.add_argument("--out", default="results/bas")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    byt = {k: load_seeds(runs, k) for k, _ in TRACKS}
    n_seeds = min(len(v) for v in byt.values())
    chance = byt["noball"][0]

    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_bas_results.tex").write_text(results_table(byt, chance, n_seeds) + "\n")
    (out / "tab_bas_per_class.tex").write_text(per_class_table(byt, chance, n_seeds) + "\n")

    # plain-text mirror, so the numbers are readable without a LaTeX run
    o = byt["noball"][0]["model"]["overall"]
    ch = chance["chance"]["overall"]
    lines = [f"OVERALL  (mean over {n_seeds} seeds, [min-max])", ""]
    lines.append(f'{"input":22} {"match":8} ' +
                 " ".join(f'{"mAP@"+str(t)+"s":>18} {"mAPw@"+str(t)+"s":>18}' for t in TOLS))
    for key, name in TRACKS:
        for m in TEST + ["overall"]:
            cells = []
            for t in TOLS:
                for k in (f"mAP@{t}s", f"mAPw@{t}s"):
                    mu, lo, hi = agg(byt[key], m, k)
                    cells.append(f"{mu:.3f} [{lo:.3f}-{hi:.3f}]" if m == "overall"
                                 else f"{mu:.3f}")
            lines.append(f'{name:22} {("Pooled" if m=="overall" else m):8} ' +
                         " ".join(f"{c:>18}" for c in cells))
    lines.append(f'{"uniform cadence":22} {"Pooled":8} ' +
                 " ".join(f'{ch[f"mAP{w}@{t}s"]:18.3f}' for t in TOLS for w in ("", "w")))
    lines += ["", "PER CLASS", "",
              f'{"class":26} {"n":>5} ' +
              " ".join(f'{f"chance@{t}s":>10} {f"traj@{t}s":>10} {f"+ball@{t}s":>11}'
                       for t in TOLS)]
    for lab in sorted(BAS_LABELS, key=lambda l: -o["support"][l]):
        n = o["support"][lab]
        cells = []
        for t in TOLS:
            cells.append(f'{ch[f"perClass@{t}s"][lab]:10.3f}')
            cells.append(f'{agg_class(byt["noball"], t, lab):10.3f}')
            cells.append(f'{agg_class(byt["ball"], t, lab):11.3f}')
        lines.append(f'{lab:26} {n:5} ' + " ".join(cells) +
                     ("   (n<30, not a measurement)" if n < LOW_SUPPORT else ""))
    txt = "\n".join(lines)
    (out / "results.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {out}/tab_bas_results.tex, {out}/tab_bas_per_class.tex, {out}/results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

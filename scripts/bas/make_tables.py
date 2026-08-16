"""Emit the paper's two BAS tables from scored predictions.

Produces LaTeX for ``tab:bas_results`` and ``tab:bas_per_class`` in
paper/sections/02_results.tex, plus a plain-text summary.

TWO REPORTING RULES ARE ENFORCED HERE, not left to whoever edits the .tex

1.  Every per-class AP is printed with its support. Header has 5 ground-truth events in the
    whole test split and Goal has 10; an AP over 5 instances takes only a few distinct
    values and is noise. The table marks those rows rather than hiding them in a caption.

2.  Every headline mAP is printed next to the chance floor. On this dataset that floor is
    not near zero: Pass occurs every 2.4 s and Drive every 2.7 s, so spots emitted at a
    fixed cadence land inside a 5 s tolerance window constantly. A support-weighted mAP@5s
    of 0.44 is what guessing achieves. Reporting a model number without it would invite the
    reader to compare against zero.

USAGE
    python scripts/bas/make_tables.py --pred outputs/bas/pred_test \\
        --baseline outputs/bas/pred_uniform --matches 128057 132831
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_utils.soccertrack_v2 import BAS_LABELS  # noqa: E402
from src.evaluation.bas_map import format_report, score_many  # noqa: E402

GT = Path("/data/share/SoccerTrack-v2/data/production/bas")
LOW_SUPPORT = 30


def fmt(v: float) -> str:
    return "--" if v != v else f"{v:.3f}"


def results_table(model: dict, base: dict | None, matches: list[str], label: str) -> str:
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{Ball action spotting on the SoccerTrack v2 test split, from "
         r"ground-truth player trajectories alone (no pixels, no ball). "
         r"$\text{mAP}$ is the macro mean over the 12 classes; $\text{mAP}_w$ weights each "
         r"class by its support. The uniform-cadence row emits spots at each class's mean "
         r"training-set rate and is the floor that guessing achieves.}",
         rf"  \label{{{label}}}", r"  \small",
         r"  \begin{tabular}{lrcccc}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{$\tau = 1\,$s} & \multicolumn{2}{c}{$\tau = 5\,$s} \\",
         r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"    Match & $n$ & mAP & mAP$_w$ & mAP & mAP$_w$ \\",
         r"    \midrule"]
    for m in matches:
        e = model[m]
        L.append(f"    {m} & {e['n_gt']:,} & {fmt(e['mAP@1s'])} & {fmt(e['mAPw@1s'])} & "
                 f"{fmt(e['mAP@5s'])} & {fmt(e['mAPw@5s'])} \\\\")
    L.append(r"    \midrule")
    o = model["overall"]
    L.append(f"    Pooled & {o['n_gt']:,} & \\textbf{{{fmt(o['mAP@1s'])}}} & "
             f"\\textbf{{{fmt(o['mAPw@1s'])}}} & \\textbf{{{fmt(o['mAP@5s'])}}} & "
             f"\\textbf{{{fmt(o['mAPw@5s'])}}} \\\\")
    if base is not None:
        b = base["overall"]
        L.append(r"    \midrule")
        L.append(f"    \\textit{{Uniform cadence (chance)}} & {b['n_gt']:,} & "
                 f"{fmt(b['mAP@1s'])} & {fmt(b['mAPw@1s'])} & {fmt(b['mAP@5s'])} & "
                 f"{fmt(b['mAPw@5s'])} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def per_class_table(model: dict, base: dict | None, label: str) -> str:
    o = model["overall"]
    b = base["overall"] if base else None
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{Per-class average precision on the SoccerTrack v2 test split, from "
         r"trajectories alone. $n$ is the number of ground-truth events; rows marked "
         r"$\dagger$ have $n < 30$, where AP takes only a few distinct values and should "
         r"not be read as a measurement. Classes are ordered by support.}",
         rf"  \label{{{label}}}", r"  \small",
         r"  \begin{tabular}{lrcccc}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{trajectory model} & \multicolumn{2}{c}{chance} \\",
         r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"    Class & $n$ & AP@1s & AP@5s & AP@1s & AP@5s \\",
         r"    \midrule"]
    order = sorted(BAS_LABELS, key=lambda l: -o["support"][l])
    for lab in order:
        n = o["support"][lab]
        mark = r"$^\dagger$" if n < LOW_SUPPORT else ""
        row = f"    {lab}{mark} & {n} & {fmt(o['perClass@1s'][lab])} & " \
              f"{fmt(o['perClass@5s'][lab])}"
        row += (f" & {fmt(b['perClass@1s'][lab])} & {fmt(b['perClass@5s'][lab])}"
                if b else " & -- & --")
        L.append(row + r" \\")
    L.append(r"    \midrule")
    L.append(f"    Macro mean (12 classes) & {o['n_gt']:,} & {fmt(o['mAP@1s'])} & "
             f"{fmt(o['mAP@5s'])}"
             + (f" & {fmt(b['mAP@1s'])} & {fmt(b['mAP@5s'])}" if b else " & -- & --")
             + r" \\")
    L.append(f"    Support-weighted mean & {o['n_gt']:,} & {fmt(o['mAPw@1s'])} & "
             f"{fmt(o['mAPw@5s'])}"
             + (f" & {fmt(b['mAPw@1s'])} & {fmt(b['mAPw@5s'])}" if b else " & -- & --")
             + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--matches", nargs="*", default=["128057", "132831"])
    ap.add_argument("--out-dir", default="outputs/bas/tables")
    a = ap.parse_args()

    model = score_many(Path(a.pred), Path(a.gt), a.matches)
    base = score_many(Path(a.baseline), Path(a.gt), a.matches) if a.baseline else None

    print("=" * 92)
    print("TRAJECTORY MODEL")
    print("=" * 92)
    print(format_report(model, a.matches))
    if base:
        print()
        print("=" * 92)
        print("UNIFORM-CADENCE CHANCE BASELINE  (same protocol, rates from the training split)")
        print("=" * 92)
        print(format_report(base, a.matches))
        print()
        print("=" * 92)
        print("MODEL MINUS CHANCE")
        print("=" * 92)
        mo, bo = model["overall"], base["overall"]
        for k in ("mAP@1s", "mAPw@1s", "mAP@5s", "mAPw@5s"):
            d = mo[k] - bo[k]
            verdict = "above chance" if d > 0 else "BELOW CHANCE"
            print(f"  {k:9}  model {mo[k]:.4f}   chance {bo[k]:.4f}   "
                  f"delta {d:+.4f}   {verdict}")
        print()
        print(f'{"class":26} {"n":>5} {"model@1s":>9} {"chance@1s":>10} {"delta":>8}   '
              f'{"model@5s":>9} {"chance@5s":>10} {"delta":>8}')
        for lab in sorted(BAS_LABELS, key=lambda l: -mo["support"][l]):
            n = mo["support"][lab]
            m1, c1 = mo["perClass@1s"][lab], bo["perClass@1s"][lab]
            m5, c5 = mo["perClass@5s"][lab], bo["perClass@5s"][lab]
            tag = "  (n<30, noise)" if n < LOW_SUPPORT else ""
            print(f"{lab:26} {n:5} {m1:9.4f} {c1:10.4f} {m1-c1:+8.4f}   "
                  f"{m5:9.4f} {c5:10.4f} {m5-c5:+8.4f}{tag}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_bas_results.tex").write_text(
        results_table(model, base, a.matches, "tab:bas_results") + "\n")
    (out / "tab_bas_per_class.tex").write_text(
        per_class_table(model, base, "tab:bas_per_class") + "\n")
    (out / "scores.json").write_text(json.dumps(
        {"model": model, "chance": base}, indent=2))
    print(f"\nwrote {out}/tab_bas_results.tex, {out}/tab_bas_per_class.tex, "
          f"{out}/scores.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

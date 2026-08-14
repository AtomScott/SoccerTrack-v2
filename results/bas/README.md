# BAS results — for the manuscript

Generated artefacts, committed so the numbers are reviewable in the diff rather than only
in someone's `outputs/`. Regenerate with:

```bash
python scripts/bas/make_tables.py \
    --pred outputs/bas_final/pred_test \
    --baseline outputs/bas/pred_uniform \
    --matches 128057 132831 --out-dir results/bas
```

| file | goes where |
|---|---|
| `tab_bas_results.tex` | replaces the `tab:bas_results` table environment in `paper/sections/02_results.tex` |
| `tab_bas_per_class.tex` | replaces the `tab:bas_per_class` table environment in the same file |
| `scores.json` | the full scored output, including per-match and per-class breakdowns for both the model and the chance baseline |

**The manuscript is deliberately not edited here.** `paper/HANDOFF_TO_CODING_AGENT.md` says
"Don't change the paper" and "Leave alone: the paper's prose. Report findings; the paper
agent writes them." Confirmed with Atom on 2026-08-14.

## Three things the prose has to say, and currently does not

1. **This is a trajectory baseline, not a video one.** `02_results.tex` currently introduces
   the BAS result as "a video backbone extracting clip features across the full-match feed";
   these numbers come from ground-truth player positions with no pixels and no ball. It is an
   upper bound assuming perfect GSR, not a deployable system.

2. **The chance floor has to appear beside the headline.** A support-weighted mAP@5s of
   0.405 is what a fixed-cadence guess achieves on this test split, because Pass occurs every
   2.4 s and Drive every 2.7 s. The `tab:bas_results` table above includes that row.

3. **The benchmark is 21,431 events, not 23,663.** 2,232 events fall in a third 45-minute
   period, in three matches, for which no video and no GSR file exists. Per Atom's decision
   they are excluded from the benchmark and from the headline count. See
   `docs/bas-findings.md` §1.

Two factual errors in the manuscript were found while doing this and are **not** fixed,
because that is Atom's call:

- `02_results.tex:226` and `04_methods.tex:148` both state `position` is milliseconds from
  kickoff of the half. It is absolute match time. Following either misaligns every
  second-half event by 45 minutes.
- `04_methods.tex` describes only the video pipeline, so there is no methods text for the
  model that produced these numbers.

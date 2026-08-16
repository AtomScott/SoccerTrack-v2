# BAS results — for the manuscript

Generated artefacts, committed so the numbers are reviewable in the diff rather than only
in someone's `outputs/`. Regenerate with:

```bash
bash /tmp/folds.sh                          # five folds x two conditions
python scripts/bas/make_cv_table.py         # tab_bas_cv.tex + tab_bas_per_class.tex
python scripts/bas/make_dataset_table.py    # tab_bas_dataset.tex
```

Two tracks, per Atom's decision to ship the ball and report both:

| file | what it is |
|---|---|
| `tab_bas_dataset.tex` | the ten matches, the split, annotated vs benchmark event counts |
| `tab_bas_cv.tex` | **the headline** — five-fold cross-match results, both input conditions |
| `tab_bas_per_class.tex` | per-class AP pooled over all five folds, i.e. all ten matches |
| `tab_bas_results.tex` | per-match breakdown on the Challenge fold |
| `paper_text.tex` | draft Methods and Results prose |
| `folds.json` | every fold's full scored output |

`noball/` and `ball/` hold the superseded single-split runs, kept only for the protocol
sensitivity comparison above. Two input conditions throughout: **trajectory** (82 features,
reproducible from the released GSR alone) and **trajectory + ball** (101 features, adding the
provider ball track).

## Headline: five-fold cross-match cross-validation

| | macro mAP@1s | wtd mAP@1s | macro mAP@5s | wtd mAP@5s |
|---|---|---|---|---|
| trajectory + ball | **0.662** ± 0.065 | **0.825** ± 0.035 | **0.701** ± 0.041 | **0.857** ± 0.014 |
| trajectory | 0.417 ± 0.028 | 0.526 ± 0.034 | 0.574 ± 0.048 | 0.754 ± 0.022 |
| uniform chance | 0.025 | 0.105 | 0.100 | 0.403 |

± is the standard deviation across the five folds. **Report this, not a single split.** The
across-fold SD is 4.3x (trajectory) and 2.3x (+ball) the across-seed SD, so which matches are
held out matters several times more than initialisation, and the Challenge pair happens to be
the weakest of the five folds for the trajectory condition (0.391 against a 0.417 mean).

`tab_bas_results.tex` gives the per-match breakdown on the Challenge fold, which must
accompany any with-ball figure: the ball is intact in 128057 (0.451 -> 0.782 macro mAP@1s)
and clamped in 132831 (0.360 -> 0.576).

**Protocol sensitivity worth knowing.** Holding the test pair fixed and changing only the
validation pair from {117093, 132877} to {117092, 117093} moves the with-ball figure from
0.519 to 0.670 -- five times the seed SD -- because 132877 is one of the clamped-ball matches.
The validation rule is therefore fixed as part of the protocol: each fold validates on the
next fold's test pair.

## Three things the prose has to say, and currently does not

1. **This is a trajectory baseline, not a video one.** `02_results.tex` currently introduces
   the BAS result as "a video backbone extracting clip features across the full-match feed";
   these numbers come from ground-truth player positions with no pixels and no ball. It is an
   upper bound assuming perfect GSR, not a deployable system.

2. **The chance floor has to appear beside the headline.** A support-weighted mAP@5s of
   0.405 is what a fixed-cadence guess achieves on this test split, because Pass occurs every
   2.4 s and Drive every 2.7 s. The `tab:bas_results` table above includes that row.

3. **The benchmark is 21,432 events, not 23,663.** 2,231 events fall in a third 45-minute
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

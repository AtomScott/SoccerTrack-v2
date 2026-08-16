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
| trajectory + ball | **0.599** ± 0.064 | **0.796** ± 0.036 | **0.666** ± 0.045 | **0.825** ± 0.028 |
| trajectory | 0.222 ± 0.021 | 0.253 ± 0.030 | 0.524 ± 0.047 | 0.673 ± 0.026 |
| uniform chance | 0.008 | 0.033 | 0.065 | 0.283 |

± is the standard deviation across the five folds, which is several times the across-seed
spread — **which matches are held out matters more than initialisation**, so report the
cross-validated mean rather than a single 8/2 draw.

**The metric is SoccerNet's, verified.** Their tolerance is a half-width (`@1s` means
±0.5 s), assignment runs from ground truth to the highest-scoring prediction in the window,
and the PR curve is sampled at 200 fixed thresholds. `tests/test_bas_map_soccernet_parity.py`
checks all three against a port of their source and requires agreement to 1e-9. An earlier
version of this package used ±1 s and prediction-first matching, which inflated every figure
by a factor of two to four.

**The two conditions separate at the tight tolerance, not the loose one.** Tightening τ from
5 s to 1 s costs the trajectory model 0.302 and the with-ball model 0.067. Player
configuration nearly suffices to say an event happened within five seconds; the ball says
when.

`tab_bas_results.tex` gives the per-match breakdown on the Challenge fold, which must
accompany any with-ball figure: the ball is intact in 128057 (0.267 → 0.734 macro mAP@1s) and
clamped in 132831 (0.191 → 0.530).

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

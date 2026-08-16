# BAS handoff to the paper-writing agent

Everything the manuscript needs for the ball action spotting half, in one place. Written
2026-08-14 from measured results; every figure here is reproducible from `scripts/bas/` and
traceable to `results/bas/folds.json`.

Nothing in `paper/` has been edited, per `paper/HANDOFF_TO_CODING_AGENT.md`. This document
says what to write, where it goes, what may safely be claimed, and — as importantly — what
must not be.

---

## 1. The headline

Five-fold cross-match cross-validation, all ten matches, SoccerNet metric.

| input | macro mAP@1s | wtd mAP@1s | macro mAP@5s | wtd mAP@5s |
|---|---|---|---|---|
| **trajectory + ball** (101 feat) | **0.599** ± 0.064 | **0.796** ± 0.036 | **0.666** ± 0.045 | **0.825** ± 0.028 |
| **trajectory** (82 feat) | 0.222 ± 0.021 | 0.253 ± 0.030 | 0.524 ± 0.047 | 0.673 ± 0.026 |
| uniform cadence (chance) | 0.008 | 0.033 | 0.065 | 0.283 |

± is the standard deviation across the five folds. Ground-truth tracks; offline (non-causal)
spotter; upper bound on any system that must estimate tracks first.

## 2. Files, and where each goes

| file | destination |
|---|---|
| `tab_bas_dataset.tex` | dataset section — the ten matches, split, event counts, actor linkage |
| `tab_bas_cv.tex` | **the headline results table** |
| `tab_bas_per_class.tex` | per-class AP pooled over all ten matches |
| `tab_bas_results.tex` | per-match breakdown on the Challenge fold |
| `paper_text.tex` | draft Methods and Results prose, ready to adapt |
| `folds.json` | every fold's full scored output, if a number needs checking |
| `results.txt` | plain-text mirror of the tables |

Supporting detail, not for the paper but for anyone verifying: `docs/bas-findings.md`.

## 3. The three claims worth making, and their evidence

**(a) Complete game state substantially determines ball events, but does not time them.**
From player positions alone, macro mAP is 0.524 at τ = 5 s and 0.222 at τ = 1 s. Adding the
ball costs only 0.067 between those tolerances where the trajectory model loses 0.302. So
player configuration nearly suffices to say *that* an event occurred within five seconds, and
the ball's contribution is almost entirely to say *when*. At τ = 1 s the ball is worth a
factor of 2.7.

**(b) Duel outcomes are not recoverable from 2-D tracking, with or without the ball.**
`Ball Player Block` reaches 0.069 without the ball and 0.181 with it; `Player Successful
Tackle` 0.046 and 0.121. Against `Pass` at 0.286 and 0.853, these are the floor in both
conditions. Both classes ask which of two converging players won a contact, and neither the
players' ground positions nor the ball's location answers that. This is a statement about the
modality, not about the model, and it is the most defensible negative result here.

**(c) A ten-match benchmark cannot rest on a single held-out pair.** The standard deviation
across the five folds is 0.021 (trajectory) and 0.064 (with ball), several times the spread
across training seeds. Changing only the *validation* pair, with test held fixed, moved the
with-ball figure by 0.15. Hence cross-validation, and hence a fixed validation rule as part
of the protocol.

## 4. Caveats that must travel with the numbers

These are not hedges; each one changes how a figure should be read.

1. **Ground-truth tracks.** Every number assumes perfect game-state reconstruction. It is an
   upper bound, not a deployable system. The predicted-track variant was not run.
2. **Offline.** Centred convolutions give each timestep ±25 s of context. Standard for action
   spotting, but it does not describe a real-time system.
3. **The chance floor is not zero.** Weighted mAP@5s of 0.283 is what a fixed-cadence guess
   achieves, because Pass occurs every 2.4 s. Never present a weighted mAP@5s without it.
4. **The with-ball figure pools two different regimes.** The ball is intact in 128057 and
   clamped to the pitch in 132831. On the Challenge fold the ball takes 128057 from 0.267 to
   0.734 and 132831 from 0.191 to only 0.530. Quote the per-match table alongside.
5. **`Header` (n = 31) and `Goal` (n = 44)** are reported for completeness and are not
   measurements even pooled over ten matches.
6. **The benchmark is 21,432 events, not 23,663.** The difference is a third period, played
   in three matches, that was never filmed or tracked.

## 5. Corrections needed in the existing manuscript

Found while doing this work, deliberately not applied.

- **`02_results.tex:226` and `04_methods.tex:148`** both state that `position` is milliseconds
  from the kickoff of the half. It is **absolute match time**. Following either misaligns
  every second-half event by 45 minutes. `docs/format-bas.md` now documents the real
  behaviour.
- **`02_results.tex:295`** introduces the BAS baseline as "a video backbone extracting clip
  features across the full-match feed". That is not the system that produced these numbers and
  would contradict the tables.
- **`04_methods.tex`** has no methods text for a trajectory-based spotter. `paper_text.tex`
  supplies one.

## 6. For the dataset section

- 1,400,774 tracked frames, 934 minutes, 30.8 M player-frames at 22 players per frame.
- Split: train 6 matches, validation 2, test 2; five-fold cross-match for the headline, with
  fold 0 fixed to the Challenge pair.
- Actor linkage is **99.5%** of benchmark events, 98.6–99.9% per match. This is a genuine
  strength and is what lets an event be joined to a player's trajectory.
- Class imbalance is **301×**; `Pass` and `Drive` are 82% of all events.
- **Three matches were played as three 45-minute periods** and only two were filmed. 2,231
  events (9.4%) have no imagery and no tracks.
- **A fully team-disjoint split is impossible** with these fixtures: 筑波大学-B appears in four
  matches, 筑波大学-C1 in three.
- **The panoramic video is not one resolution**: 4096×1080, 4096×1084, 3840×1906, 3840×1504.

### Data defects worth documenting in the release, not hiding

- **The ball is clamped to the pitch rectangle in 132831 and 132877** — exactly zero frames
  off-pitch against 2.1–3.2% elsewhere — and their `ballStatus` is a constant `INPLAY` where
  the other eight carry `BALLOUT/HOME/AWAY/NEUTRAL`. 132831 is in the test split. The cost is
  measured, not merely asserted: the ball is worth +0.47 macro mAP@1s on 128057 and +0.34 on
  132831, and `Out` reaches only 0.364 even with the ball because two matches cannot show the
  event that defines the class.
- **The BAS event clock differs from the tracking clock by one second on 14 of the 20 halves.**
  The split is 6 halves at 0 and 14 at ~25 frames, with nothing in between — the same 6/14
  split the GSR side measured for video frame offsets. Correcting it was worth 0.12 mAP.
- **`visibility` is documented in the format spec and populated on none of the 23,663 events.**

## 7. What was not done, and why

- **Video-based BAS (Experiment A) is not attempted, and on this data it is probably not the
  right experiment.** Frames are 4K covering the whole pitch, so a model input of 224–398 px
  means either a ~10× downscale, at which the ball is well under a pixel, or a crop. A
  ball-centred crop from annotation is an oracle; a crop from a detected ball needs a ball
  detector that does not exist, since the released game state has no ball. The honest options
  are a play-centred crop from player positions or a full-frame downscale as a floor, and
  neither is likely to beat the trajectory result reported here. Recommend framing this as
  future work with the reason stated, rather than as a gap.
- **Experiment B on predicted tracks** requires GSR predictions across all ten matches, which
  do not exist yet. This is the one experiment that would let the paper say whether GSR or BAS
  is the limiting factor end to end, and it is the highest-value follow-up.
- **Nothing in the manuscript has been edited.**

## 8. Reproducing

```bash
python scripts/bas/extract_tracks.py --all --workers 5      # stream GSR -> positions
python scripts/bas/extract_tracks.py --validate
python scripts/bas/measure_t0.py --all --write              # tracking-clock alignment
python scripts/bas/extract_ball.py --all                    # ball, all ten matches
python scripts/bas/measure_event_offset.py --validate       # event-clock alignment
python scripts/bas/measure_event_offset.py --all --write
python scripts/bas/audit_annotations.py --check-alignment --paper-stats
python scripts/bas/build_dataset.py --workers 6
python scripts/bas/build_dataset.py --workers 6 \
    --ball  /data/share/SoccerTrack-v2/data/derived/bas/ball \
    --out   /data/share/SoccerTrack-v2/data/derived/bas/dataset_ball
bash /tmp/rerun.sh                                          # five folds x two conditions
python scripts/bas/make_cv_table.py
python scripts/bas/make_dataset_table.py
python -m pytest tests/ -q                                  # 36 tests
```

## 9. Where the numbers came from, and where they were wrong first

Recorded because the paper should be able to state that the protocol was verified rather than
assumed, and because several figures in earlier drafts of this handover were wrong.

- **The metric was not SoccerNet's.** Ours used a ±τ window, prediction-first matching and a
  per-prediction PR curve; theirs uses ±τ/2, ground-truth-first matching and 200 fixed
  thresholds. Ours read 0.047 where theirs read 0.013 on random data.
  `tests/test_bas_map_soccernet_parity.py` now checks against a port of their source to 1e-9.
  Every number was recomputed and every model retrained, because the decoder and stopping
  epoch had been selected against the wrong objective.
- **`t0` was wrong on 18 of 20 halves** before the event-clock measurement, 14 of them by a
  full second.
- **The third-period count** was reported as 2,225, then 2,232, before settling at **2,231**.
  Only the last is consistent with the 21,432-event benchmark.
- **The prior rule-based detector required three separate fixes** before it produced a
  meaningful number at all — see §10.

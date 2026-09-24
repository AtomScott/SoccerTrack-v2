# Structural priors on the 45-minute half — what worked, what backfired, and why

The game supplies exact constraints, verified against this half's ground truth:

- **exactly 22 detections in every frame** (min 22, median 22, max 22)
- **exactly 11 per team per frame** (both sides, median and max 11)
- **23 identities** over the match: 12 left (1 GK + 11), 11 right (1 GK + 10) — one substitution
- **jersey numbers unique within a team**
- **no referees at all** in this match

Unconstrained clustering throws all of that away: the pipeline's team clustering produces a
45.9/54.1 split where the truth is 50/50 by construction.

Priors were applied **post-hoc to the finished 45-minute predictions**, so each variant costs
minutes instead of 37 hours of GPU, and every number below is a real GS-HOTA delta on the full
half against the 18.094 baseline.

## Results

| variant | GS-HOTA | DetA | AssA | vs baseline |
|---|---|---|---|---|
| baseline (no priors) | **18.094** | 14.672 | 22.319 | — |
| exactly 11 per team, per **frame** | 16.350 | 14.701 | 18.189 | **−1.74** |
| jersey unique within team | 16.424 | 10.711 | 25.185 | **−1.67** |
| both together | 14.674 | 10.857 | 19.835 | **−3.42** |

**Every one of them made the score worse.** That is the interesting part.

## Why: per-detection accuracy is the wrong objective

The 11-per-team prior **improved per-detection team accuracy from 76.17% to 82.86%** — measured on
36,683 confident geometric matches — and still cost 1.74 GS-HOTA. The mechanism is visible in the
breakdown: DetA was unchanged (14.672 → 14.701) while **AssA fell from 22.319 to 18.189**.

Deciding each frame independently lets a player's team label flip between frames. Counted directly:
**124,186 label flips inside tracklets.** GS-HOTA partitions detections into classes by
`(role, team, jersey)`, so a flip does not merely mis-label a detection — it moves that detection
into a different class, breaking the track. The metric punishes that far harder than it rewards
being right more often.

The jersey prior shows the mirror image. Forcing distinct numbers within a team **improved AssA
(22.319 → 25.185)** because it made identities more consistent, but **collapsed DetA (14.672 →
10.711)** because tracklets with weak evidence were assigned confident-looking wrong numbers, and
every such detection lands in a class with no ground-truth counterpart.

## The lesson, stated for reuse

> A structural prior only helps if it is imposed at the level the metric evaluates. GS-HOTA
> evaluates **tracks**, so a per-frame constraint — however correct on average — can reduce the
> score by breaking temporal consistency. And a hard constraint applied where the evidence is weak
> converts uncertainty into confident error, which the class partition punishes without mercy.

Two corollaries worth carrying into any further work:

1. **Optimise consistency, not accuracy.** Per-detection attribute accuracy and GS-HOTA can move in
   opposite directions. Do not tune against the former.
2. **Constraints need an abstention path.** Forcing an assignment where evidence is absent is worse
   than leaving the attribute null.

## Reproduce

`scratchpad/mem/apply_priors_45.py` builds the variants from the saved state and prediction file;
`scripts/gsr/score_one.py` scores each. Track-level variants (one team per tracklet, and a
confidence-thresholded jersey assignment) are the natural follow-ups and are measured separately.

---

# Round 2: applying priors at the track level, and where association actually breaks

## The team prior works when applied to tracks

| variant | GS-HOTA | DetA | AssA | vs 18.094 |
|---|---|---|---|---|
| 11-per-team, per **frame** | 16.350 | 14.701 | 18.189 | −1.74 |
| **11-per-team → one team per tracklet** | **19.724** | 16.567 | 23.493 | **+1.63** |
| one team per tracklet, decided once | 19.295 | 15.962 | 23.333 | +1.20 |

Same prior, same information, **a 3.37-point swing from placement alone**, and both components
improve (DetA 14.67 → 16.57, AssA 22.32 → 23.49). Per-frame balance produced 124,186 team-label
flips inside tracklets; the track-level variants produce zero by construction.

**19.724 is the best GS-HOTA measured on this half.**

## Jersey cannot be rescued by constraints

| variant | GS-HOTA | DetA | AssA |
|---|---|---|---|
| jersey unique within team (any support threshold) | 17.569 | 11.375 | **27.140** |
| same, support ≥ 100 | 17.230 | 10.789 | 27.518 |
| null the weak ones instead | 19.724 | 16.567 | 23.493 |

Uniqueness produces the **largest AssA gain seen anywhere (+4.8)** and still loses overall, because
DetA falls further. And nulling weak jerseys changes nothing at all — the pipeline already nulls
exactly those tracklets. The reason is sparsity: **median per-tracklet jersey support is 0.0**, and
75% of tracklets carry no jersey evidence whatsoever. There is nothing to constrain.

## Why association could not be repaired post hoc

Re-grouping the 347 tracklets into 58 identities — appearance-based merging under a hard
temporal-exclusion constraint, verified to be using the new ids — moved AssA from **13.409 to
13.410**. Two formulations were tried (greedy closest-pair merging, and largest-first slot
colouring); both land at 57–58 groups, because tracklets are sparse frame *sets* with gaps rather
than contiguous intervals, so the conflict graph is not an interval graph and needs far more colours
than the 24-identities-per-frame maximum would suggest.

The null result is explained by measuring what the tracklets actually contain:

| measure | value |
|---|---|
| mean tracklet purity (detection-weighted) | **55.8%** |
| median distinct ground-truth identities **inside one tracklet** | **13** |
| median predicted tracklets **per real player** | **31** |
| largest single tracklet's share of one player | commonly 20–45% |

**Predicted tracklets are not fragments of one player; they are mixtures of about thirteen.** A
player is simultaneously scattered across ~31 tracklets. Re-grouping whole tracklets therefore
cannot help — the units being grouped are themselves wrong, and would have to be split.

This also bounds the two aggregation tricks that did help: per-tracklet voting for team pools
evidence over ~13 different people, which is why it buys a few points and no more.

## Consequences for the work list

- **Association must be fixed in the online tracker**, not afterwards. AssA 13.4 is the dominant
  ceiling (fixing it alone would take the attainable score from 26.0 to 71.6, against 7.94 for all
  attributes combined), and no post-hoc re-association with the available ReID features touches it.
- **Do not spend more effort on jersey.** Three quarters of tracklets have no evidence, and the
  constraint that most improves consistency costs more in detection than it returns.
- **The track-level team prior is worth keeping**: +1.63, cheap, and it composes with anything else.

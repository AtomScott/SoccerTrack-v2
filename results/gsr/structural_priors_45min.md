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

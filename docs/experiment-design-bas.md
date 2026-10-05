# BAS experiment design — SoccerTrack v2

Ball Action Spotting: predict the time and class of 12 ball-event types across a match. This
document is the plan for review, not a record of results. Nothing here has been run yet.

---

## 0. Blocker: the evaluator does not rank by confidence

**This must be fixed before any BAS baseline is run, or the first numbers will be wrong in a way
that looks fine.**

`src/evaluation/bas_map.py::_ap_tolerant` states its contract in a comment:

> Predictions with scores could be sorted by score; we have no scores here, so we treat input
> order as ranked (callers: sort by confidence desc).

Its own caller violates that contract. `_map_per_class` does:

```python
p = sorted((e for e in pred if e.label == label), key=lambda e: (e.half, e.t_ms))
```

which re-sorts predictions **by time**, discarding any confidence ordering. Average precision is
defined over a confidence-ranked list; traversing in time order produces a precision-recall curve
with no meaning. A detector that emits many low-confidence spots is not penalised in ranking, and
one that ranks its output perfectly gains nothing.

`Event` also has no confidence field (`half, clock, t_ms, label, team, player_id, visibility`),
so there is currently nowhere to put a score.

**Why the existing test does not catch it:** `tests/test_bas_map_identity.py` scores ground truth
against itself. Every prediction is then a true positive, so AP is 1.0 in any order. This is the
same structural blind spot that hid three defects in the GS-HOTA prediction path — an identity
test validates the metric's configuration, never the path a real prediction takes.

**Fix required:** add `score: float` to `Event`, sort by it descending in `_map_per_class`, and
add a test that a *deliberately mis-ranked* perfect prediction scores **worse** than a
well-ranked one. If that test cannot fail, it is not testing ranking.

---

## 1. What is measured

**Metric.** Temporal mean average precision at tolerance windows, per the SoccerNet BAS protocol:
**mAP@1s** (tight) and **mAP@5s** (loose), plus a per-class breakdown. A prediction is a true
positive when an unmatched ground-truth event of the same class exists in the same half within
the tolerance. Report both, because the two answer different questions: mAP@5s asks "did the
system notice the event", mAP@1s asks "did it locate it precisely".

**The 12 classes.** Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player
Block, Player Successful Tackle, Free Kick, Goal.

**Time alignment — resolved, and it contradicts the docs.** `docs/format-bas.md` states that
`position` is milliseconds from kickoff *of the half*. It is not: it is **absolute from the start
of the match**. Verified on 117093, whose half-2 events run 2,700,760–5,506,280 ms with
`gameTime` "2 - 45:00" rather than restarting near zero. `Event.t_ms_in_half` subtracts a nominal
45-minute half to recover the per-half offset; that is approximate, because real halves run over
with stoppage. **For frame-exact alignment use the per-period `frameStart` in
`<match>_tracker_box_metadata.xml`, not the nominal subtraction.**

Given what a one-second misalignment cost the GSR baseline — GS-HOTA 18.79 vs 37.16 on the same
data — this must be verified against imagery before any BAS number is trusted, not assumed from
arithmetic. `scripts/gsr/measure_frame_offset.py` establishes the video-to-label offset per half
and applies directly.

**Two further divergences from the documentation**, both already handled in the loader: the event
array is keyed `actions`, not `annotations`; and labels are UPPER CASE in the files while the docs
and paper use Title Case. Three matches (117092, 132831, 132877) carry a third block of events
whose `gameTime` omits the half prefix and whose `position` runs past 5,400,000 ms.

**First measurement to run, before any modelling:** the actual class distribution and event count
per match. The class imbalance drives everything downstream — Pass and Drive will dominate while
Goal may have single-digit support, and a macro-averaged mAP over 12 classes is extremely
sensitive to the rare ones. Do not design the split before seeing these counts.

---

## 2. Experiment A — video-based spotting

**Model.** T-DEED, or a newer temporally-dense spotting architecture. The task is precise temporal
localisation, which is what these are built for; generic video classifiers backboned on Kinetics
are a poor fit because they optimise clip-level labels rather than frame-level onsets.

**The input problem, which is the hard part.** Frames are 4096×1080 covering the whole pitch. A
model input of 224–398 px means either downscaling by ~10× — at which the ball is well under a
pixel and the events are unrecognisable — or cropping. Options, in order of honesty:

1. **Ball-centred crop from ground truth.** Clean and simple, but it is an **oracle**: it uses
   annotation at inference time. If used, it must be labelled an oracle in the paper and reported
   separately, never as the headline.
2. **Ball-centred crop from a detected ball.** Legitimate, but requires a ball detector we do not
   have — the GSR ground truth contains no ball at all (20 outfield + 2 goalkeepers, verified).
3. **Play-centred crop from predicted player positions.** The centroid of detected players tracks
   play closely and uses no annotation. This is the defensible default.
4. **Full-frame downscale.** Honest and cheap; likely to fail on small-ball events. Worth running
   once as a floor, precisely because it establishes what resolution costs.

**Recommendation:** (3) as the primary, (1) as an oracle upper bound, (4) as a floor. The gap
between (1) and (3) is itself a result about how much of the task is "finding the ball".

**Weights.** Pretrained action-spotting weights exist for SoccerNet broadcast footage. Panoramic
fixed-camera footage is a different domain — different scale, no camera cuts, no replays — so
expect zero-shot transfer to be poor, and treat a vanilla run as a floor rather than a baseline.

---

## 3. Experiment B — trajectory-based spotting

**Input.** Player positions on the pitch over time, from GSR. No pixels. Features per frame:
positions, velocities, inter-player distances, convex-hull area, centroid motion, and possession
proxies such as which player is nearest the play centroid.

**Model.** A temporal model over the per-frame feature sequence — a TCN or a small transformer —
predicting per-frame class logits, decoded to spots by peak-picking with non-maximum suppression.

**The decision that must not be blurred: ground-truth tracks or predicted tracks?**

- **Ground-truth tracks** measure "is the game state sufficient to infer events" — a genuine
  scientific question, and one this dataset is unusually well suited to answer, because it has
  complete 22-player positions on every frame.
- **Predicted tracks** measure the deployable pipeline, and inherit every GSR error.

Both are worth running, and **the difference between them is the interesting result**. What is not
acceptable is reporting one and describing it as the other.

**Recommendation:** ground-truth tracks as the primary result (it is the question the dataset is
positioned to answer), predicted tracks as the realistic system, both reported.

Note the strong prior: many of the 12 classes are probably *not* inferable from positions alone.
Header versus Pass is a body-part distinction invisible in 2D foot positions. Expect trajectory
BAS to do well on Out, Throw In, Free Kick and Goal (which have distinctive spatial signatures)
and poorly on Header, Ball Player Block and Drive. A per-class breakdown is therefore not optional
— the aggregate mAP would hide exactly the structure that makes this interesting.

---

## 4. What each experiment shows, and what it does not

| | shows | does NOT show |
|---|---|---|
| A, play-centred crop | whether events are recognisable from panoramic video at practical resolution | anything about a ball detector we do not have |
| A, oracle crop | an upper bound given perfect ball localisation | deployable performance |
| B, ground-truth tracks | whether complete game state determines events | deployable performance |
| B, predicted tracks | the realistic end-to-end system | which of GSR or BAS is the limiting factor, unless A and B are compared |

---

## 5. Compute

BAS is far cheaper than GSR. Trajectory models (B) train on position sequences in minutes on CPU
or a single GPU — the whole dataset is roughly 1.4M player-frames per half, which is small.

Video models (A) need frames. The GSR staging already extracts them at ~110 GB per half, and
`scripts/gsr/stage_soccernetgs.py` produces them; crops for BAS can reuse the same `img1`
directories rather than re-extracting.

The GPU is currently committed to the GSR scaling runs (~87 h). **Experiment B needs no GPU and
can run in parallel**; Experiment A should be queued behind GSR.

---

## 6. Open questions for Atom

1. **Split.** The SoccerTrack Challenge 2025 split assigns 128057 and 132831 to test. Does BAS use
   the same split as GSR? Using the same one keeps the paper coherent; a different one would need
   justifying.
2. **Is trajectory BAS in scope for this paper**, or is the video path enough? B is cheap and is
   the more novel claim, but it is a second system to build and validate.
3. **Ground-truth versus predicted tracks for B** — my recommendation is both, primary on ground
   truth. Confirm.
4. **The oracle crop.** Acceptable as a clearly-labelled upper bound, or would you rather it not
   appear at all?
5. **`visibility: "not shown"` events.** Some events are annotated whose ball contact is not
   observable in frame. Are these excluded from evaluation, kept, or reported separately? They are
   unfair to a video model and fair to a trajectory model, so the choice interacts with the
   comparison in section 4.

# BAS findings — SoccerTrack v2

What was measured while building the ball-action-spotting baseline, in the order it changed a
decision. Everything here is reproducible from the scripts named; nothing is inferred from
arithmetic alone where it could be measured instead.

Companion to [`experiment-design-bas.md`](experiment-design-bas.md), which is the plan.
This is the record.

---

## 1. Three matches have a third 45-minute period, and nothing was filmed for it

`117092`, `132831` and `132877` were played as **three** 45-minute periods. The evidence is
independent of the event files:

- `132831` and `132877` declare `matchFullTime="8100000"` — 135 minutes — in
  `<match>_tracker_box_metadata.xml`.
- `117092` declares `matchFullTime="5400000"` but carries a fourth XML element,
  `<period period="EXTRA_FIRST_HALF" frameStart="251" frameEnd="68501"
  matchTimeStart="5400000" matchTimeEnd="8130000">`, i.e. a real 45.5-minute frame range.
- `132831`'s raw folder holds `132831_segment_0.mp4`, `_segment_1.mp4` **and** `_segment_2.mp4`.

**2,232 events — 9.4% of the 23,663 annotated — fall in that third period, and no third
video and no third GSR file exist for any of the three matches.** Only
`<match>_panorama_{1st,2nd}_half.mp4` and `<match>_{1st,2nd}.json` were released.

Per match, as a share of that match's annotations:

| match | annotated | period 1 | period 2 | period 3 | period 3 share |
|---|---|---|---|---|---|
| 117092 | 3,142 | 1,029 | 1,057 | **1,056** | 33.6% |
| 132831 **(test)** | 3,162 | 1,250 | 1,191 | **721** | 22.8% |
| 132877 | 2,731 | 1,183 | 1,095 | **453** | 16.6% |
| the other seven | 14,628 | 7,482 | 7,144 | 2 | 0.0% |
| **total** | **23,663** | **10,944** | **10,487** | **2,232** | **9.4%** |

The two stray period-3 events among the other seven matches are single events in 118576 and
118577 that land just past the end of their half's tracking. One event is not a period; the
`--paper-stats` period count requires a block of at least 50 before calling a period played.

The consequence is not cosmetic. Scoring a *perfect* period-1-and-2 prediction against the
released files gives **mAP@1s 0.8409 instead of 1.0000**, and the entire penalty lands on
132831 (0.7576) while 128057 is untouched (1.0000). Any model evaluated on the test split
would have been docked ~16 macro mAP points for annotations it cannot see, by a different
amount on each match.

Per Atom's decision the third period is outside the benchmark and outside the headline
count. **The benchmark is 21,431 events.**

## 2. Neither field in the event file determines the period on its own

`gameTime` is `"<period> - <ABSOLUTE mm:ss>"` — the clock does not restart at each half, so a
second-half event reads `"2 - 45:00"`. That much was already known. What is new:

**The period prefix is unreliable on exactly the three-period matches.** Of the 2,232
third-period events, 2,129 carry no prefix at all and **103 carry a prefix of `1` or `2` beside
a clock past 90 minutes** — for example `{"gameTime": "1 - 135:27", "position": "8127120"}`.
Trusting the prefix puts those events 135 minutes into a 45-minute half.

**`position` cannot replace it, because the periods overlap on the nominal clock.** 118576's
first half runs to 48:29 while its second half starts at 45:00; the two ranges genuinely
intersect. Any rule of the form `position // 45min` therefore misassigns real stoppage-time
events. Measured overlap, per match, in the first half's overrun past 45:00: 118576 3.5 min,
118577 3.1 min, 118575 2.9 min, 118578 1.9 min, 128058 0.9 min.

**The rule adopted** (`src/data_utils/bas_periods.py`) takes the prefix as a hypothesis and
accepts it only when the resulting frame index lands inside that period's annotated GSR frame
range — i.e. only when the event actually has input data. That is the property the benchmark
depends on, and it keeps the 19 prefix-`2` events past 90 minutes that really are
second-half stoppage while rejecting the 96 that are not.

## 3. The event-to-track time mapping, measured rather than computed

    frame = 1 + (position - t0_ms) / 40
    t0_ms = matchTimeStart - (frameStart - 251) * 40

**GSR GameState frame 1 corresponds to raw tracking frame 251 in every half.** This was
established by sweeping the alignment against the per-match pitch-plane CSVs under
`data/interim`, which exist for 117092 and 117093 and were produced by a different pipeline:

| candidate mapping | mean abs. error over 110 positions |
|---|---|
| frame 1 = the period's first raw frame | 0.1135 m |
| **one frame later** | **0.0000 m** |
| two frames later | 0.1325 m |

An exact zero over 110 positions, at a 1.05 m coordinate quantisation, is not a coincidence.
The same check pins the metre scaling and the field-extraction regex at the same time, which
is why it is the validation mode of `scripts/bas/extract_tracks.py` rather than a one-off.

It also explains the null `bbox_pitch` values: the three halves whose `frameStart` is 252
rather than 251 (117092 1st, 118578 2nd, 128058 2nd) have a leading GameState frame with no
source row behind it, and those are exactly the halves with a non-zero null-pitch count.

132831 and 132877 declare **no `<period>` elements at all**, so their `t0` falls back to the
nominal 0 / 2,700,000 ms. Across the eight matches that do declare it, `t0` never departs
from nominal by more than 33 ms — under one frame — and the check in §4 confirms the fallback
independently.

## 4. Alignment and actor linkage confirmed against a randomised null

A falsifiable test that consults neither annotation's clock: **the taker of a `Throw In` must
be standing on a touchline**, so their |y| should be near 34 m on a 105×68 pitch. The
statistic is the fraction of takers with |y| > 32 m, compared against a null of 60
displacements of 20–120 s in both directions.

**All twenty halves clear their own null, at z = 9.7 to 29.4.** This validates the time
mapping *and* the event-to-actor linkage in one measurement, including for the two matches
using the nominal fallback (132831: z = 20.2 / 21.0; 132877: z = 13.3 / 9.7).

Seven matches score 0.84–1.00 in absolute terms. The three three-period matches score lower
(117092 0.62/0.67, 132831 0.52/0.68, 132877 0.36/0.75) because **their tracking is
systematically compressed**: the 99th percentile of |x| is 47–50 m against 52–54 m elsewhere,
on an identically declared 105×68 pitch, so their players never quite reach the lines. That
is a property of the tracking, not of the mapping.

### A sharper estimator was tried and rejected

Cross-correlating collective player speed against play-stopping (`Out`) and play-restarting
(`Throw In`) events **peaked at +18, +41 and +26 frames on three halves whose true offset is
known to be 0**, with z of only 1.7–1.9. It is measuring how long players take to coast to a
stop after the whistle — a real property of football, not an annotation offset. Recorded so
it is not re-attempted.

## 5. Pitch coordinates are quantised to 1.05 m, which dictates the features

The positions derive from a normalised pitch coordinate stored to two decimal places, so x
lands on multiples of **1.05 m** and y on multiples of **0.68 m**. A one-frame finite
difference therefore carries 1.05 / 0.04 = **26 m/s of pure quantisation noise**. Measured
median apparent player speed against window length, on 128057's first half:

| window | 0.08 s | 0.24 s | 0.40 s | 0.96 s | 1.60 s |
|---|---|---|---|---|---|
| median measured speed | 0.00 | 2.83 | 1.70 | 1.42 | 1.43 m/s |
| quantisation-only prediction | 13.12 | 4.38 | 2.62 | 1.09 | 0.66 m/s |

At 0.08 s the median is *exactly zero* — the position has not changed — while the 95th
percentile is 13.13 m/s, exactly one step. Velocity only becomes physical near 1 s. The
feature builder therefore supplies velocity at two scales (0.48 s and 1.52 s) rather than
picking one.

### The same quantisation broke the ball proxy

The GSR ground truth contains **no ball at all** (20 outfield players + 2 goalkeepers), so
the best available proxy for where the ball is is the closest opposing pair. Computed with
`argmin`, that pair is **exactly tied on 6.7% of frames**, and the tie-break then chooses
arbitrarily between equally valid pairs — arbitrary jitter in the single most important
feature group.

This surfaced from `tests/test_bas_augment.py`, which mirrors the raw tracks, rebuilds the
features through the ordinary path, and requires a cheap feature-space mirror to agree: all
nine contest-derived columns disagreed under an x-mirror while every other column matched,
which is not what an index bug looks like. Replacing `argmin` with a soft minimum over all
opposing pairs (weights `exp(-d/2m)`) took the worst relative disagreement from 3.1e-01 to
9.5e-07.

## 6. `visibility` is documented but never populated

`docs/format-bas.md` documents `"visible"` / `"not shown"`. The field is present on **0 of
23,663 events**. Open question 5 of the experiment design — whether to exclude, keep or
separately report `"not shown"` events — is moot as the data stands.

## 7. Class imbalance, and what it does to a macro mean

301× between the largest and smallest class in the benchmark. Actor-link coverage is high
across the board, which is a genuine strength — an event that names its actor can be joined
to that player's track.

| class | benchmark n | share | actor-linked | test-split n |
|---|---|---|---|---|
| Pass | 9,316 | 43.47% | 99.5% | 1,928 |
| Drive | 8,257 | 38.53% | 99.6% | 1,704 |
| High Pass | 1,157 | 5.40% | 99.8% | 226 |
| Out | 771 | 3.60% | 100.0% | 161 |
| Cross | 394 | 1.84% | 100.0% | 56 |
| Throw In | 385 | 1.80% | 100.0% | 83 |
| Ball Player Block | 353 | 1.65% | 92.9% | 62 |
| Player Successful Tackle | 307 | 1.43% | 100.0% | 50 |
| Shot | 266 | 1.24% | 100.0% | 57 |
| Free Kick | 150 | 0.70% | 100.0% | 31 |
| Goal | 44 | 0.21% | 100.0% | **10** |
| Header | 31 | 0.14% | 93.5% | **5** |
| **total** | **21,431** | | **99.5%** | **4,373** |

**Header has 5 ground-truth events in the whole test split and Goal has 10.** An AP over 5
instances takes only a handful of distinct values. A 12-class macro mean gives that noise the
same weight as Pass, which has 1,928. Per Atom's decision both the macro mean and a
support-weighted mean are reported, always labelled, and no per-class figure is printed
without its `n`.

## 8. The chance floor is not near zero

Pass occurs every 2.4 s of tracked play and Drive every 2.7 s, so spots emitted at a fixed
cadence land inside a 5 s tolerance window constantly. Measured on the **validation** matches
with per-class rates taken from the **training** split only:

| baseline | macro mAP@1s | weighted mAP@1s | macro mAP@5s | weighted mAP@5s |
|---|---|---|---|---|
| uniform cadence | 0.0311 | 0.1167 | 0.1115 | **0.4413** |
| random times | 0.0248 | 0.1062 | 0.1050 | 0.3574 |

**A support-weighted mAP@5s below 0.441 is worse than guessing.** Any headline BAS figure has
to be read against this, not against zero. `scripts/bas/baseline_priors.py` also provides an
`oracle-rate` variant that peeks at the test counts; it is labelled as not a legitimate
baseline and exists only to show how much of a score is attributable to knowing the class
frequencies.

## 9. What the model is and is not sensitive to

All figures below are **validation** (117093, 132877). The test split was scored once, after
the configuration was fixed.

### Capacity and regularisation barely matter

| configuration | val mAP@1s | best epoch |
|---|---|---|
| hidden 64, dropout 0.4, **lr 1e-3** | **0.3893** | 15 |
| hidden 128, dropout 0.2 | 0.3753 | 8 |
| hidden 64, dropout 0.2 | 0.3739 | 16 |
| hidden 96, dropout 0.3 | 0.3736 | 11 |
| hidden 64, dropout 0.4 | 0.3695 | 8 |
| hidden 128, dropout 0.4 | 0.3642 | 19 |

A 2× range in width and a 2× range in dropout move the result by 0.025. Halving the
learning rate helps more than either. This is not a model-capacity-limited problem.

### Reflection augmentation is worth +0.025, measured against its own control

`hidden 64, dropout 0.4` with augmentation scores **0.3695**; the identical configuration
with `--no-augment` scores **0.3446**. Same width, same dropout, same seed, same schedule —
the only difference is the mirroring. Without it the model overfits by epoch 6.

### Validation loss and validation mAP disagree, and the metric wins

With `pos_weight` up to 50 the BCE is dominated by confident false positives on the rare
classes. In the first full run validation *loss* rose monotonically from epoch 3 while
validation *mAP* kept improving to epoch 8. Selecting on loss would have discarded the
better detector. Model selection is on mAP throughout.

### Per-team shape and motion carry almost all of the signal

Feature-group ablation, validation only. Groups are **zeroed rather than removed**, so every
run has the same input width and the same parameter count — a group that does not matter
cannot be confused with a smaller model.

| features kept | columns | val mAP@1s | share of full |
|---|---|---|---|
| all four groups | 82 | **0.3893** | 100% |
| team only | 24 | **0.3844** | **98.7%** |
| everything except contest | 69 | 0.3595 | 92.3% |
| occupancy only | 36 | 0.2649 | 68.0% |
| contest only | 13 | 0.2587 | 66.5% |

**24 of the 82 columns get 98.7% of the result.** The `team` group is each side's centroid,
centroid velocity, dispersion, convex hull, mean and max speed, and defensive/attacking
line — nothing about the ball proxy. The explicit contest point and the 36-column occupancy
grid together add 0.005 on top of it.

The two single-group runs are the more interesting comparison. **Occupancy alone contains no
velocity feature at all** — it is a static player-density grid — and still reaches 68%,
because the temporal model derives motion from the sequence of grids itself. "No velocity
features" is not the same as "no motion information" once a TCN is reading 25 s of context.

An earlier version of this ablation was **wrong and had to be redone**. `L_ct_dist` and
`R_ct_dist` — each team's distance to the contest point — were grouped with `team` because of
their names, so the "everything except contest" run still saw two contest-derived scalars and
could not have supported the claim it existed to test. Groups are now defined by provenance
and partition all 82 columns exactly. The corrected `team_only` (0.3844, genuinely
contest-free) is barely different from the flawed one (0.3874), so the conclusion survives —
but it survived by luck, not by construction.

### The decoding floor is always driven to its minimum

Every configuration chose `floor=0.02, nms_rows=5` from a 6×4 grid. That is a property of
11-point interpolated AP, which takes the maximum precision at each recall level: appending
lower-ranked predictions can raise recall and can never lower the score. The SoccerNet
protocol places no cap on prediction count, so this is legitimate rather than a trick — but
it means **the number of emitted spots must be reported next to the score**, which
`format_report` does. The chosen setting emits roughly 3 spots per ground-truth event.

## 10. The result on the test split

**Scored once**, on 128057 and 132831, with the configuration (hidden 64, dropout 0.4,
lr 1e-3, decode floor 0.02, NMS 5 rows, epoch 15) fixed beforehand on the validation
matches. Ground-truth tracks — this is an upper bound that assumes perfect GSR, not a
deployable pipeline.

| | macro mAP@1s | weighted mAP@1s | macro mAP@5s | weighted mAP@5s |
|---|---|---|---|---|
| **trajectory model** | **0.3155** | **0.4136** | **0.5203** | **0.7126** |
| uniform cadence (chance) | 0.0255 | 0.1065 | 0.0945 | 0.4054 |
| margin over chance | +0.2900 | +0.3071 | +0.4259 | +0.3072 |

Per match: 128057 macro mAP@1s 0.3388, 132831 0.3378. The pooled figure (0.3155) is lower
than either because pooling scores each class on all of its instances at once rather than
averaging two separate handfuls. 20,291 spots were emitted against 4,373 ground-truth
events, roughly 4.6 predictions per event.

### Per class, against chance

| class | n | AP@1s | chance | Δ | AP@5s | chance | Δ |
|---|---|---|---|---|---|---|---|
| Pass | 1,928 | 0.433 | 0.149 | +0.284 | 0.751 | 0.491 | +0.260 |
| Drive | 1,704 | 0.427 | 0.102 | +0.325 | 0.756 | 0.470 | +0.286 |
| High Pass | 226 | 0.412 | 0.004 | +0.408 | 0.578 | 0.067 | +0.511 |
| Out | 161 | 0.369 | 0.006 | +0.363 | 0.577 | 0.037 | +0.540 |
| Throw In | 83 | 0.354 | 0.023 | +0.331 | 0.674 | 0.023 | +0.651 |
| Ball Player Block | 62 | **0.035** | 0.000 | **+0.035** | **0.089** | 0.008 | **+0.081** |
| Shot | 57 | 0.337 | 0.007 | +0.330 | 0.496 | 0.007 | +0.489 |
| Cross | 56 | 0.479 | 0.000 | +0.479 | 0.604 | 0.005 | +0.598 |
| Player Successful Tackle | 50 | **0.072** | 0.011 | **+0.061** | **0.120** | 0.023 | **+0.097** |
| Free Kick | 31 | 0.276 | 0.003 | +0.273 | 0.577 | 0.003 | +0.573 |
| Goal † | 10 | 0.399 | 0.000 | +0.399 | 0.788 | 0.000 | +0.788 |
| Header † | 5 | 0.193 | 0.000 | +0.193 | 0.236 | 0.000 | +0.236 |

† support below 30; these APs take only a few distinct values and are not measurements.

### Two caveats that belong next to these numbers

**This is an offline spotter, not a real-time one.** The TCN uses centred convolutions, so
each timestep sees ±25 s of context, and the long-scale velocity feature spans ±0.76 s.
Both look into the future. That is standard for action spotting — the SoccerNet protocol
scores a whole match offline — but it means these figures do not describe what a live
system could do.

**It assumes perfect tracks.** Every number here is on ground-truth GSR positions. A
deployable system has to estimate those first and inherits every GSR error, so this is an
upper bound. The design document's Experiment B calls for both variants and the gap between
them as the result; only the ground-truth half is done, per Atom's decision to defer the
predicted-track variant.

### What the per-class pattern says

**Two classes fail almost completely.** Ball Player Block (AP@1s 0.035) and Player
Successful Tackle (0.072) are barely above chance at either tolerance. Both are duel
*outcomes* — who won a contact — and that is a body-level distinction with no signature in
2 D foot positions quantised to 1.05 m. Two players converging looks the same whether the
tackle succeeds, the block happens, or neither.

**Out and Shot are detected but not timed.** Out goes 0.369 → 0.577 and Shot 0.337 → 0.496
between the 1 s and 5 s tolerances, much steeper than Pass (0.433 → 0.751 from a far higher
base). Both are events defined by *where the ball goes*, not by what a player does: players
only react afterwards, so the trajectory tells you it happened without pinning when.

**Two of the design document's predictions were wrong.** `experiment-design-bas.md` §3
expected trajectory BAS to "do well on Out, Throw In, Free Kick and Goal ... and poorly on
Header, Ball Player Block and **Drive**". Ball Player Block and Header were right, but
Drive is one of the better classes (+0.325 over chance) and Out is the *weakest* of the four
predicted to be strong at the tight tolerance. Recorded because the prediction was written
before anything was run, and it was half right.

**Per-class figures move a lot between splits**, on two test matches. Throw In is 0.510 on
validation and 0.354 on test; Out is 0.283 on validation and 0.369 on test. The
class-level ordering is stable, the individual values are not, and nothing below n≈50
should be quoted as a point estimate.

## 11. Incidental findings, for whoever needs them

- **The released videos are not all 4096×1080.** 117092 is 3840×1906, 132831 is 3840×1504,
  132877 is 4096×1084, and the remaining seven are 4096×1080. Relevant to any crop strategy
  for the video experiment.
- **117092's GSR file declares `height: 1504, width: 3840`** for its images while its video
  is 3840×1906. Those are 132831's dimensions. Not load-bearing for trajectory work, which
  reads only `bbox_pitch`, but it means image-space work on 117092 cannot trust the header.
- **The interim pitch-plane CSVs contain a ball track**, under `id: "ball"`, for 117092 and
  117093 only. The released GSR files contain no ball. It is not usable for the benchmark —
  neither test match has one — but it means ball tracking data exists at source for at least
  some matches, which bears on the video experiment's crop options.
- **Tracked play totals 1,400,874 frames across the twenty halves** — 934 minutes, or
  30,819,228 player-frames at 22 entities per frame. This is measured over periods 1 and 2
  only, and may help settle the paper's "900 minutes vs 1.62 million frames" contradiction.
- **`132831` and `132877` have 550–875 null `bbox_pitch` records per half**, an order of
  magnitude more than any other match, scattered rather than at the start.

---

## Reproducing

```bash
python scripts/bas/extract_tracks.py --all --workers 5      # 54 GB streamed, ~2 min
python scripts/bas/extract_tracks.py --validate            # vs the independent CSVs
python scripts/bas/audit_annotations.py --write-periods --check-alignment --paper-stats
python scripts/bas/build_dataset.py --workers 6
python -m pytest tests/test_bas_map_periods.py tests/test_bas_augment.py -q -s
```

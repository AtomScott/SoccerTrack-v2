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

**2,231 events — 9.4% of the 23,663 annotated — fall in that third period, and no third
video and no third GSR file exist for any of the three matches.** Only
`<match>_panorama_{1st,2nd}_half.mp4` and `<match>_{1st,2nd}.json` were released.

Per match, as a share of that match's annotations:

| match | annotated | period 1 | period 2 | period 3 | period 3 share |
|---|---|---|---|---|---|
| 117092 | 3,142 | 1,029 | 1,057 | **1,056** | 33.6% |
| 132831 **(test)** | 3,162 | 1,249 | 1,191 | **722** | 22.8% |
| 132877 | 2,731 | 1,183 | 1,095 | **453** | 16.6% |
| the other seven | 14,628 | 7,482 | 7,146 | 0 | 0.0% |
| **total** | **23,663** | **10,943** | **10,489** | **2,231** | **9.4%** |

These counts shifted slightly when `t0` was corrected (§3): an event within a frame or two of
a period boundary can cross it. Earlier drafts of this document quoted 2,225 and then 2,232;
2,231 is the figure from the current measured period table, and is the only one consistent
with the 21,432-event benchmark that the dataset build reports.

The consequence is not cosmetic. Scoring a *perfect* period-1-and-2 prediction against the
released files gives **mAP@1s 0.8409 instead of 1.0000**, and the entire penalty lands on
132831 (0.7576) while 128057 is untouched (1.0000). Any model evaluated on the test split
would have been docked ~16 macro mAP points for annotations it cannot see, by a different
amount on each match.

Per Atom's decision the third period is outside the benchmark and outside the headline
count. **The benchmark is 21,432 events.**

## 2. Neither field in the event file determines the period on its own

`gameTime` is `"<period> - <ABSOLUTE mm:ss>"` — the clock does not restart at each half, so a
second-half event reads `"2 - 45:00"`. That much was already known. What is new:

**The period prefix is unreliable on exactly the three-period matches.** Of the 2,231
third-period events, 2,129 carry no prefix at all and **102 carry a prefix of `1` or `2` beside
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
second-half stoppage while rejecting the 102 that are not.

## 3. The event clock, and the two clocks I conflated

Two different questions hide in `frame = 1 + (position - t0_ms)/40`:

**(a) which GSR frame a raw tracking `frameNumber` denotes.** Answered exactly by
`scripts/bas/measure_t0.py`, which aligns released GSR player positions against
`<match>_tracker_box_data.xml`. At the correct alignment the two sources *are the same data*,
so the residual must be **zero**, not merely small: all 20 halves give exactly 0.0000 m with
a clean V (0.21–0.29 m one frame either side). Two things only this could find — 132831 and
132877 number frames from 1 rather than 251, and `n_frames` had been taken from
`info.seq_length`, which those two overstate by 25.

**(b) which GSR frame a BAS `position` denotes.** A *different* quantity, because the events
are a separate annotation pass. I answered (b) with (a) and it was wrong.

### What (b) actually is: 14 of 20 halves are one second out

`scripts/bas/measure_event_offset.py` asks a question the ball makes possible and that has no
semantic ambiguity: **is the annotated actor the player nearest the ball?** 99.5% of events
name their actor, and at a ball event that actor has the ball — whether the annotation marks
the first touch or the release.

It is sharp (peak 0.86–0.95 against a 0.13–0.25 floor) and it validates: planted shifts of
−20/+20/+40 frames are recovered exactly. The readings are **strictly bimodal**:

| shift | halves |
|---|---|
| ~0 frames | **6** — 117092/1st, 117093/2nd, 118575/1st, 118575/2nd, 118577/1st, 118578/2nd |
| ~+25 frames (1.00 s) | **14** — all the rest |

Nothing lands in between. A semantic lag would vary continuously; a clean split at exactly
one second is a bookkeeping slip — and it is **the same 6/14 split the GSR side measured for
video frame offsets**, so the same per-half slip surfaces in both tasks.

`t0` was therefore wrong on **18 of 20 halves**: 14 by a full second, 4 by ≤80 ms.

### Three anchors that failed first, and why

- **Throw-in touchline** (§4). Flat across ±2 s, because a thrower stands on the line for
  seconds either side. It catches a gross error — which is what it was built for — and it
  passed at z = 13–21 on halves that were a second out.
- **Ball speed step at a strike.** The estimator validates, but the annotated moment need not
  be the moment of contact, so it confounds a clock offset with an unknown semantic lag. Its
  readings were bimodal 0/25 across halves whose `t0` was independently verified.
- **Ball crossing the pitch boundary at an `Out`.** Peak hit rate only 0.05–0.43, and exactly
  0.00 for 132831 and 132877 — because their ball is clamped to the pitch (§11).

The lesson is the one the GSR half already paid for, in a new disguise: verifying source A
against source B does not license assuming source C shares B's clock.

## 4. Alignment and actor linkage confirmed against a randomised null

A falsifiable test that consults neither annotation's clock: **the taker of a `Throw In` must
be standing on a touchline**, so their |y| should be near 34 m on a 105×68 pitch. The
statistic is the fraction of takers with |y| > 32 m, compared against a null of 60
displacements of 20–120 s in both directions.

**All twenty halves clear their own null, at z = 9.7 to 29.4**, which confirms the
event-to-actor linkage and rules out gross misalignment.

**It does not confirm the timing, and §3 is why.** A thrower stands on the line for seconds
either side of the throw, so this statistic is flat across ±2 s and passed comfortably on the
fourteen halves that were a full second out. It was built to catch a gross error and it did
exactly that; treating it as evidence of fine alignment was my mistake, not the test's. The
sharp instrument is the actor-nearest-ball anchor in §3.

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
| **total** | **21,432** | | **99.5%** | **4,373** |

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

Scored on 128057 and 132831, **three seeds**, with the configuration fixed on validation
beforehand (hidden 64, dropout 0.4, lr 1e-3, decode floor 0.02, NMS 5 rows). Ground-truth
tracks; offline spotter; upper bound assuming perfect GSR.

| | macro mAP@1s | weighted mAP@1s | macro mAP@5s | weighted mAP@5s |
|---|---|---|---|---|
| **with ball** (101 feat) | **0.662** ± 0.065 | **0.825** ± 0.035 | **0.701** ± 0.041 | **0.857** ± 0.014 |
| **no ball** (82 feat) | 0.417 ± 0.028 | 0.526 ± 0.034 | 0.574 ± 0.048 | 0.754 ± 0.022 |
| uniform cadence (chance) | 0.025 | 0.105 | 0.100 | 0.403 |

± is the standard deviation across the five cross-match folds, which is **4.3× (trajectory)
and 2.3× (+ball) the standard deviation across training seeds**. Which matches are held out
matters several times more than initialisation. The Challenge pair is the weakest of the five
folds for the trajectory condition, at 0.391 against a mean of 0.417.

**The validation pair matters even more than the test pair.** Holding test fixed at the
Challenge matches and changing only validation from {117093, 132877} to {117092, 117093} moves
the with-ball figure from 0.519 to **0.670** — five times the seed SD — because 132877 is one
of the two clamped-ball matches, so a with-ball model tuned on it selects a decode threshold
suited to censored data. An earlier version of this document reported 0.487 as the with-ball
headline for exactly that reason; it was a protocol artefact, not a property of the model.

### What the ball is worth depends entirely on whether it is intact

| test match | ball | no ball → with ball, macro mAP@1s |
|---|---|---|
| 128057 | intact | 0.472 → **0.769**  (+0.30) |
| 132831 | **clamped to the pitch** | 0.365 → 0.357  (−0.01) |

On the match whose ball is real the ball is worth +0.30 macro mAP@1s and +0.35 weighted. On
the match whose ball is censored (§11) it is worth nothing, and at the 5 s tolerance it is
actively **harmful** (0.527 → 0.450) — the model learned to rely on features that behave
differently there. The pooled figure averages these two regimes and should not be read as
"what the ball buys".

### Per class, AP@1s

| class | n | chance | no ball | with ball | Δ |
|---|---|---|---|---|---|
| Pass | 1,928 | 0.154 | 0.496 | 0.826 | +0.330 |
| Drive | 1,703 | 0.093 | 0.480 | 0.750 | +0.270 |
| High Pass | 226 | 0.008 | 0.403 | 0.522 | +0.120 |
| Out | 161 | 0.006 | 0.411 | 0.465 | +0.054 |
| Throw In | 83 | 0.023 | 0.461 | 0.527 | +0.065 |
| Ball Player Block | 62 | 0.000 | **0.046** | **0.107** | +0.061 |
| Shot | 57 | 0.005 | 0.446 | 0.717 | +0.271 |
| Cross | 56 | 0.000 | 0.589 | 0.535 | −0.054 |
| Player Successful Tackle | 50 | 0.011 | **0.168** | **0.099** | −0.070 |
| Free Kick | 31 | 0.003 | 0.256 | 0.315 | +0.058 |
| Goal † | 10 | 0.000 | 0.731 | 0.822 | +0.091 |
| Header † | 5 | 0.000 | 0.450 | 0.545 | +0.095 |

† support below 30; these take only a few distinct values and are not measurements.

### What the per-class pattern says

**Duel outcomes stay unsolved even with the ball.** Ball Player Block (0.107) and Player
Successful Tackle (0.099) are the two worst classes in both tracks, and the tackle gets
*worse* when the ball is added. Both are questions about **who won a contact**, and knowing
where the ball is does not answer that — two players converge on it either way. This is the
clearest statement in the result of what 2-D tracking cannot represent.

**The ball buys the most where timing is the difficulty.** Pass (+0.33), Drive (+0.27) and
Shot (+0.27) gain most: the player configuration already says an event is happening, and the
ball says exactly when. Consistent with that, the ball adds nothing at the 5 s tolerance
overall (0.565 → 0.550) — at 5 s the configuration alone already suffices.

### Two caveats that belong next to these numbers

**This is an offline spotter.** Centred convolutions give each timestep ±25 s of context, and
the long-scale velocity spans ±0.76 s. Both look into the future. Standard for action
spotting, which is scored offline over a whole match, but it does not describe a live system.

**It assumes perfect tracks.** Every number is on ground-truth GSR positions, so it is an
upper bound on any system that has to estimate them. The predicted-track variant is deferred.

## 11. Incidental findings, for whoever needs them

- **The released videos are not all 4096×1080.** 117092 is 3840×1906, 132831 is 3840×1504,
  132877 is 4096×1084, and the remaining seven are 4096×1080. Relevant to any crop strategy
  for the video experiment.
- **117092's GSR file declares `height: 1504, width: 3840`** for its images while its video
  is 3840×1906. Those are 132831's dimensions. Not load-bearing for trajectory work, which
  reads only `bbox_pitch`, but it means image-space work on 117092 cannot trust the header.
- **The ball exists for ALL TEN matches** in `<match>_tracker_box_data.xml`, on every frame,
  with zero missing values — alongside per-player provider-computed `speed` and a per-frame
  `ballStatus`. The released GSR annotations contain no ball; the raw tracking does.
  `scripts/bas/extract_ball.py` puts it in the same reference frame as the released players
  (y as-is for all ten, measured by which choice puts the ball at somebody's feet: 2.0–2.6 m
  from the nearest player on eight matches, 2.3–4.1 m on the other two).

- **But the ball is CENSORED on 132831 and 132877**, and one of them is a test match:

  | | eight matches | 132831, 132877 |
  |---|---|---|
  | ball x range | ±57.8 m | **±52.5 m exactly** |
  | ball y range | up to ±38.1 m | **±34.0 m exactly** |
  | frames with ball off-pitch | 1.3–6.9% | **0.00%** |
  | `ballStatus` | BALLOUT/HOME/AWAY/NEUTRAL | **constant `INPLAY`** |

  Clamped to the pitch rectangle to the millimetre, so the ball can never show a ball going
  out. This is measurable in the result: the ball is worth +0.30 macro mAP@1s on 128057 and
  nothing on 132831 (§10).

- **`ballStatus` must not be used as a feature.** `BALLOUT` is close to a direct label for the
  `Out` class on eight matches, and is a constant on the two where it would be needed. It is
  extracted so the leak can be quantified, and excluded from every feature set.
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
python scripts/bas/measure_t0.py --all --write            # period table (t0 measured)
python scripts/bas/extract_ball.py --all                   # ball, all ten matches
python scripts/bas/measure_event_offset.py --all --write   # event clock, per half
python scripts/bas/audit_annotations.py --check-alignment --paper-stats
python scripts/bas/build_dataset.py --workers 6
python -m pytest tests/test_bas_map_periods.py tests/test_bas_augment.py -q -s
```

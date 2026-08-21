# GSR baseline results — handoff for the paper

Everything the manuscript needs from the GSR side, with the caveats attached. Written to be read
once and turned into text and tables. **No manuscript files were touched.**

Source data for every number: `results/gsr/` (CSV + JSON + chart), and the two analysis write-ups
`results/gsr/attribute_ablation_45min.md` and `results/gsr/structural_priors_45min.md`.

---

## 1. The headline result

**A full 45-minute half runs end to end and scores GS-HOTA 18.094.** Match 128057, 1st half,
67,625 frames, 1,388,671 detections, ~37 hours of compute in two resumable stages.

This is the first complete half the pipeline has produced. Before this work it could not finish
10 minutes.

### Table 1 — accuracy against sequence length (paper-ready)

Nested prefixes of **identical footage**, one fixed configuration throughout (jersey gates
`min_roi_area=100`, `min_obb_aspect_ratio=0.6`, detection-level team clustering). The only variable
is how much footage the global modules must reconcile at once.

| length | frames | detections | GS-HOTA | attrs off | tracklets | wall |
|---|---|---|---|---|---|---|
| 30 s | 750 | 16.5k | 37.155 | — | 26 | 4 min |
| 1 min | 1,500 | 33k | **47.589** | — | 35 | 12 min |
| 2 min | 3,000 | 66k | 39.882 | — | 42 | 37 min |
| 5 min | 7,500 | 165k | 30.974 | — | 60 | 2.6 h |
| 10 min | 15,000 | 301k | 25.604 | 33.908 | 99 | 5.5 h |
| **45 min** | **67,625** | **1,389k** | **18.094** | 26.036 | 347 | ~37 h |

Chart: `results/gsr/length_curve.png` (three panels — score, attribute accuracy, tracklet counts).

**The claim to make:** accuracy peaks at one minute and falls monotonically thereafter with no sign
of levelling; a full half scores **38% of the one-minute figure on the same footage**. This is the
paper's argument for why full-length evaluation is not the same task as clip-level evaluation.

---

## 2. Where the loss is (Table 2 candidate)

Each attribute switched on in turn on the full half. GS-HOTA partitions detections into classes by
`(role, team, jersey)`, so a wrong attribute is **no match at all**, not partial credit.

| configuration | GS-HOTA | DetA | AssA |
|---|---|---|---|
| geometry + association only | **26.036** | 51.209 | 13.409 |
| + role | 25.637 | 49.358 | 13.480 |
| + team | 22.364 | 35.116 | 14.374 |
| + jersey — **official** | **18.094** | 14.672 | 22.319 |

Marginal cost: **role −0.40, team −3.27, jersey −4.27.**

Attribute accuracy at 45 min, on 36,665 confident geometric matches (≤2 m): **role 97.92%, team
76.17%, jersey 45.95%.** Jersey is *wrong*, not *absent* — only 3.4% of predictions carry a null.

### The structural point worth making explicitly

`26.036` **is the ceiling for the official metric**: perfect attributes partition without losing
matches, so all attribute work combined is worth at most **7.94 points**. The ceiling itself is
`sqrt(DetA × AssA)`:

| lever | current | ceiling if perfect |
|---|---|---|
| detection | 51.2 | 36.6 |
| **association** | **13.4** | **71.6** |

Detection and localisation are respectable (DetA 51.2, LocA 84.7) — the pipeline finds players and
places them correctly. **Association is the dominant limitation by a wide margin.**

---

## 3. Why association fails — the mechanism (this is the most citable finding)

Tracklet *count* is a poor proxy and should not be quoted alone: 347 tracklets against 23 players
sounds like 15×, but **282 of them hold 0.2% of all detections**. About 65 carry the rest.

What actually characterises the failure, measured against ground truth:

| measure | value |
|---|---|
| mean tracklet purity (detection-weighted) | **55.8%** |
| median distinct GT identities **inside one tracklet** | **13** |
| median predicted tracklets **per real player** | **31** |

**Predicted tracklets are mixtures of about thirteen people, not fragments of one.** Each real
player is simultaneously scattered across ~31 tracklets.

This was confirmed by a null result: re-grouping the 347 tracklets into 58 identities under a hard
temporal-exclusion constraint moved AssA from **13.409 to 13.410**. Two formulations were tried
(greedy closest-pair merging; largest-first slot colouring) and both stall near 57 groups, because
tracklets are sparse frame *sets* with gaps, so the conflict graph is not an interval graph.
Regrouping whole tracklets cannot work — the units are themselves impure and would have to be split.

**Likely proximate causes, identified but NOT yet validated** (see §6):
- BoT-SORT runs with `track_buffer: 30` — a player unmatched for **1.2 s at 25 fps** becomes a new
  identity. Soccer occlusions routinely exceed that.
- The tracker's association ReID weights are `clip_duke.pt` — CLIP trained on **DukeMTMC pedestrian
  surveillance**. Teammates in matching kit are close to the worst case for such a model.

---

## 4. What was tried to improve the score, and what it bought

The game supplies exact structural priors, all verified true of this half's ground truth: **exactly
22 detections per frame, exactly 11 per team, 23 identities, jersey unique within team, no
referees**. The pipeline violates them (its team split is 45.9/54.1 where truth is 50/50).

| intervention | Δ GS-HOTA |
|---|---|
| **11-per-team prior applied per TRACK** | **+1.63 → 19.724** |
| one team per tracklet, decided once | +1.20 → 19.295 |
| identity re-grouping (347→58) | +0.001 |
| junk-tracklet filter (min length) | +0.014 |
| jersey uniqueness within team | −1.67 |
| 11-per-team prior applied per FRAME | −1.74 |
| both priors together | −3.42 |

**19.724 is the best measured on this half** (+9% relative), obtained post-hoc with no retraining.

### The finding that is more interesting than the improvement

The *same prior with the same information* swings **3.37 points** depending on whether it is applied
per frame or per track. Per-frame balancing **raised per-detection team accuracy from 76.17% to
82.86% and lost 1.74 GS-HOTA**, because it produced **124,186 team-label flips inside tracklets** —
and a flip moves a detection into a different class, breaking the track.

> **A structural prior only helps if imposed at the level the metric evaluates.** GS-HOTA evaluates
> tracks, so per-detection attribute accuracy and GS-HOTA can move in opposite directions. And a
> hard constraint applied where evidence is absent converts uncertainty into confident error: jersey
> uniqueness produced the largest AssA gain seen anywhere (+4.8) and still lost overall.

Jersey is not rescuable by constraint: **median per-tracklet jersey support is 0.0**; three quarters
of tracklets carry no jersey evidence at all.

Also refuted (do not re-try): temporal windowing of team clustering — fixed windows at 1/5/10/15 min
and a confidence-adaptive variant all scored **worse** than one global fit (55–68% vs 76.20% team
accuracy), so appearance drift is not the limitation. Kit-colour features gave crisp clusters
(separation 2.3–4.2 vs 0.63) but only +1.45 team accuracy, because they separate *illumination*
rather than kit.

---

## 5. Engineering findings — for Methods / Limitations, and for reproducibility

Full detail with measurements in `docs/gsr-tracklab-patches.md`. Every fix is behaviour-preserving
unless stated.

**Making a full half possible at all** (each of these independently blocked it):

| defect | effect | fix |
|---|---|---|
| `OptimizationConsistency.build_track_evidence` iterated column *values*, not unique ids | O(N²); silent for 2h22m on 5 min; **25–45 h projected at 45 min** | grouped pass, **18,920× faster, bit-identical** |
| GTA tracklet distance materialised three n×n GPU tensors for a scalar | **54.88 GB at 45 min on a 16 GB GPU** | exact separable identity, **198× less memory** |
| kernel OOM at 61% of PRTReID | 35.5 GB resident, frame only 0.23 GB | glibc arena retention; `malloc_trim` reclaimed **58.2 GB** over the run |
| `merge_dataframes` quadratic in sequence length | crippling beyond a few minutes | de-quadratified, identical GS-HOTA |

**Worth a limitations sentence:**
- **Non-finite pitch coordinates grow with length**: 0.16% of predictions at 10 min, **0.435% at
  45 min**. Calibration degrades near the horizon as sequences lengthen. These detections cannot be
  matched in pitch space at all and are dropped before scoring (count recorded in each
  `score_*.json`).
- **TrackLab's evaluator deadlocked** after inference completed — three processes blocked on futexes
  with frozen CPU counters for 12h39m. Predictions were already written, so scoring was done offline
  with `scripts/gsr/score_one.py`. Any reported number for a long sequence should be assumed to come
  from that path.
- **Orphaned dataloader workers**: children of an OOM-killed parent are reparented to init and never
  reaped; twelve were found holding 8.79 GB a day later. Relevant to anyone reproducing on a shared
  machine.
- Only **41 referee detections in 1.38M**, and this match's ground truth contains **no referees** —
  so referee handling is effectively untested here, and the 97.92% role accuracy is dominated by
  players.

---

## 6. Caveats — what must NOT be claimed

1. **Single match, single half, no error bars.** 128057 is a *test-split* match; nothing was tuned
   on it. Do not generalise the absolute numbers.
2. **Two team-clustering configurations cross over**: stock tracklet-level is better below 2 minutes,
   detection-level better by 5. A sweep that mixes them measures the configuration, not the length.
   Table 1 uses one configuration throughout — say so.
3. **The +1.63 improvement is post-hoc**, applied to saved predictions, not a pipeline change. It has
   not been validated by an end-to-end rerun.
4. **The `track_buffer` and Duke-ReID hypotheses in §3 are untested.** A sweep (30/90/250) was
   running at handoff; results are not in. Do not present them as causes, only as candidates.
5. `use_spatial_connect` was fixed from pixel to metre units (a genuine bug — pixel distance is
   perspective-dependent), but **merged zero tracklets in practice**, so it changes nothing.
6. Attribute accuracy at 5 min in panel 2 of the chart comes from an earlier measurement that may
   have used a different matching tolerance; the 45-minute figures are the trustworthy ones.

---

## 7. Open items

- `track_buffer` sweep (30/90/250) at 10 min — in flight at handoff, `scratchpad/logs/buffer_sweep.log`
- Swapping the tracker's association ReID from `clip_duke.pt` to something kit-appropriate — untried
- Four exact reductions from the module scan (`GKRoleAssignment`, `generate_tracklets`,
  `add_removed_detections`, `np.isin` on string `image_id`) — worth ~3–4 h of a 25 h run, not accuracy
- Fixing non-finite pitch coordinates at source in calibration
- BAS task — separate handoff, `docs/experiment-design-bas.md`

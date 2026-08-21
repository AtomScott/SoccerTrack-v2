# GSR pipeline: the complete experiment log

Everything tried on the GSR baseline, 2026-08-08 → 2026-08-21, in one place — wins, losses, and
null results, each with its numbers and where they came from. Companion to
[handoff-gsr-for-paper.md](handoff-gsr-for-paper.md) (which carries only what the paper can claim);
this file is the lab notebook. Machine-readable version of every entry:
`results/gsr/experiment_inventory.json` (109 records, extracted from the primary files).

**Reference sequence throughout:** match 128057, 1st half (test split), 25 fps, nested prefixes of
identical footage. Nothing was tuned on it; gate tuning used 117093 (valid split).

---

## 0. Scoreboard — the headline number over time (1-minute clip unless stated)

| date | event | GS-HOTA |
|---|---|---|
| 08-11 | first real predictions scored at all (after 3 evaluator fixes) | ~15 (misaligned) |
| 08-12 | **1-second label/pixel misalignment found and fixed** | 14.996 → 33.471 |
| 08-13 | jersey gates relaxed 500/0.6 → 100/0.6 | 33.471 → **57.824** |
| 08-15 | 5-min: detection-level team clustering | 19.484 → 30.974 |
| 08-17 | 10-min full pipeline (first ever to finish) | 25.604 |
| 08-20 | **45-min full half (first ever)** | **18.094** |
| 08-21 | + track-level team prior (post-hoc) | **19.724** |

---

## 1. Evaluation infrastructure (had to exist before any experiment)

### 1.1 GS-HOTA prediction path — three defects, fixed *(shipped, PR #25)*
The converter could score GT-vs-GT but had never scored a real prediction file: `bbox_pitch`
carried 2 of the 6 keys the pitch-space scorer reads; `image_id` was an int where the scorer calls
`len()`; predictions numbered frames 0-based against 1-based GT string ids, so **nothing ever
matched**. Post-fix: identity test HOTA 1.000000, id-mapping coverage 67,625/67,625 frames. The
prior "verification" was structurally incapable of catching this — it compared labels to labels.

### 1.2 `score_one.py` — offline scoring tolerant of bad points *(shipped)*
Needed twice: trackeval crashes on any NaN pitch coordinate (`matrix contains invalid numeric
entries`), and TrackLab's in-run evaluator **deadlocked** after the 45-min inference (3 processes
on futexes, CPU counters frozen, 12h39m). Drops unmatchable detections and reports the count.

### 1.3 Abstention is punished by the metric *(rejected as a lever)*
Nulling predicted jersey scores **6.19** against a 37.155 baseline; nulling team scores **0.00**.
A null maps to a class the GT doesn't contain, so confidence-thresholding attributes is not a
lever; only genuine accuracy is. (`attr_results.json`)

---

## 2. Released-data defects and alignment (for the dataset paper's own account)

### 2.1 The 1-second misalignment *(shipped; the single biggest fix)*
Labels lagged imagery by 25 frames. Full-window effect at every length: 30s 18.791→37.155,
1min 14.996→33.471, 5min 8.848→14.409 (attrs-off 41.1→77.4 at 30s). Pre-fix scores kept in
`sweep_scores.misaligned.json.bak`.

### 2.2 Per-half offsets are NOT uniform *(shipped)*
Box-occupancy estimator over 60 probe frames, all 20 halves: **6 halves at ~0, 14 at 24–30
frames** (`frame_offsets.csv`). 132831 estimates (147/301) are low-confidence (z 1.7/2.5) — staged
at 25 by policy; flagged. Two earlier estimator designs failed their own validation: background
subtraction gave 67 vs known 25 (kickoff players stand still, polluting the median background);
`cap.set(CAP_PROP_POS_FRAMES)` is not frame-accurate on these files and injects its own offset.

### 2.3 Released-file defects (documented)
Declared 3840×1504 vs real 4096×1080 frames; `info.id`="1" in all 20 files with colliding names;
GT annotates ~25 more frames than the video has; `docs/format-gsr.md` describes a flat format the
files aren't; calibrated keypoints shipped for 117093 only. Staging clamp bug (offset must shrink
the window: `end = min(end, n_img_gt − offset)`) found via a crashed run on 132877.

---

## 3. The length-degradation result (core science)

### 3.1 Final curve, one fixed config (gates 100/0.6 + detection-level team)
| | 30s | 1min | 2min | 5min | 10min | 45min |
|---|---|---|---|---|---|---|
| GS-HOTA | 37.155 | **47.589** | 39.882 | 30.974 | 25.604 | **18.094** |
| attrs off | — | — | — | — | 33.908 | 26.036 |
| tracklets (GT 22–23) | 26 | 35 | 42 | 60 | 99 | 347 |

With stock tracklet-level team instead: 40.352 / 57.824 / 44.810 / 19.484 — **the two configs
cross over** (stock wins ≤2 min, detection-level wins ≥5 min). Chart:
`results/gsr/length_curve.png`.

### 3.2 The common-window control (the cleanest evidence)
Score only the **identical first 750 frames** out of runs of different length:
- attrs OFF: 77.446 / 77.381 / 77.626 (30s/1min/5min runs) — **flat**. Detection, localisation,
  association on the same frames don't care how long the run was.
- attrs ON: 37.155 / 37.155 / **16.343** — attribute quality on the *same frames* collapses when
  the run is longer, because attributes are assigned globally per tracklet.

This separates "longer runs see harder content" from "the global modules degrade": it's the latter.

### 3.3 Why minute 1 is worse than minute 0 (same run, split windows, attrs off)
0–30s: 40.98 (DetRe 50.76); 30–60s: 20.84 (DetRe 30.87). LocA falls only 5.8 while DetRe falls
19.9 — detections don't vanish, they land outside the 5 m tolerance. Static calibration is the
suspect: this match's residual at its own 65 keypoints is median 2.295 m, **p95 7.770 m**, against
a 5 m matching tolerance.

### 3.4 Attribute ablation at 45 min (marginal costs)
role −0.40, team −3.27, jersey −4.27; **attributes-off (26.036) is the ceiling** for the official
metric, so all attribute work combined ≤ 7.94 points. Ceiling = √(DetA·AssA); fixing association
alone (13.4) would raise it to 71.6, vs 36.6 for perfect detection. **Association is the lever.**

### 3.5 Match difficulty is real
117093 vs 128057 at 30 s (misaligned-era, like-for-like): 29.14 vs 18.79, LocA 92.15 vs 77.13.
Per-match numbers in the paper need this spread acknowledged.

---

## 4. Per-stage experiments

### Detection / cost profile *(informational)*
750-frame profile (RTX 4060 Ti): ViTPose 10:31 (56%), RF-DETR 1:36, PRTReID 0:51, PARSeq 0:27,
rest <0:10 each. GPU stages project to ~20 h/half.

### Pose
- **ViTPose model size** (huge/base/small): GS-HOTA identical 57.824, runtime identical — the pose
  stage is not model-bound *(neutral; small is free if wanted)*.
- **Stride** (1/2/5/10): 57.824 / 52.114 / 50.426 / 50.426 — −5.7 points for a 2× stage speedup
  *(rejected: sub-2× overall for that loss)*.
- **Ankle-joint ground anchor** instead of box bottom-middle: localisation 1.139→1.357 m, **19%
  worse** — the box bottom (~5.5 px below ankles) is closer to ground contact *(rejected before
  shipping; measured first)*.

### Calibration
- **Metre-space homography objective**: −2.9 / −8.1 GS-HOTA on the two clips tried; default
  reverted to pixel *(rejected, module kept as negative result)*.
- **Non-finite pitch coordinates grow with length**: 0.16% of predictions at 10 min → **0.435% at
  45 min** (6,002 dropped). Unfixed at source *(open)*.

### Jersey
- **Diagnosed as a funnel, not a model defect**: of 31,107 detections, 3.0% get a region proposed,
  1.6% read at conf ≥0.8; 8 of 65 tracklets ever get a reading; median torso region **12×11 px**.
  The `min_roi_area=500` gate passed 6.9% of detections.
- **Gate relaxation 500/0.6 → 100/0.6** *(shipped; biggest single accuracy win)*: 1min
  33.471→57.824 (+24.35), 2min +12.54, 30s +3.20. Selected on 117093 (valid split), where the same
  direction holds (13.572→20.709).
- **At 45 min, jersey is beyond configuration**: 45.95% accurate where read, but **median
  per-tracklet jersey support is 0.0** — 75% of tracklets carry no evidence. Uniqueness-within-team
  (Hungarian over evidence): largest AssA gain seen anywhere (+4.8) but DetA collapses; net
  −1.67 to −2.15 at any support threshold *(rejected)*.

### Team
- **Detection-level clustering module** *(shipped, conditional)*: 5 min team accuracy 51.6%→86.6%,
  GS-HOTA 19.484→**30.974** (+11.49). Worse at ≤2 min (crossover, §3.1).
- **Temporal windowing** — fixed 1/5/10/15-min windows *and* confidence-adaptive (close a window on
  separation drop): **all worse than global**, 55–68% vs 76.20% team accuracy. Drift is not the
  limiter; centroid-stitching errors dominate *(rejected — including the "smart windowing" idea)*.
- **Kit-colour features** (CIELAB torso median): separation 2.3–4.2 vs ReID's 0.63, accuracy only
  +1.45 (78.68% vs 77.23%) — colour separates *illumination*, not kit *(rejected)*.
- **Structural priors** — see §5.

### Tracking / association (GTA and BoT-SORT)
- **GTA `connect_track` crashed on contact**: read `.role`/`.jersey_number` that `Tracklet` never
  carries — `use_spatial_connect` had *never actually run*. Dead code removed *(shipped)*.
- **Pixel thresholds on metre coordinates**: distance gate compared image pixels against 100.
  Rewritten in pitch metres with a 9 m/s velocity gate *(shipped)* — but with it on, connect merged
  **0 tracklets** at 2 min (42→42); GS-HOTA unchanged *(neutral in practice)*.
- **GTA region defect**: `avg_box_params.image_width: 1920` on 4096-wide frames collapses 65% of
  the frame into one region. **Fixing it scored −1.62** (config5: team-widthfix 29.353 vs 30.974) —
  the miscalibrated filter was accidentally conservative *(fix rejected on measurement)*.
- Tracklet-consolidation variant: 29.351, also −1.62 *(rejected)*. GTA merge error grows with
  length (+0 at 30 s, +5 at 1 min, +11 at 5 min) and discards a stable ~13% of detections.
- **Post-hoc identity re-grouping (347→58 under temporal exclusion)**: AssA 13.409→**13.410**.
  Two formulations (greedy merge, slot colouring) both stall at 57–58 — tracklets are sparse frame
  sets, not intervals *(null result, decisive)*.
- **Tracklet purity — the mechanism**: mean purity **55.8%**; a substantial tracklet contains a
  median of **13 distinct GT identities**; each player is scattered across a median of **31
  tracklets**. Tracklets are *mixtures*, not fragments — association must be fixed online, in the
  tracker *(the central diagnostic finding)*.
- **Untested candidates** *(in flight)*: `track_buffer: 30` = 1.2 s ID memory at 25 fps (sweep
  30/90/250 running); tracker association ReID is `clip_duke.pt` (pedestrian surveillance) —
  teammates in identical kit are its worst case.

---

## 5. Structural priors (22 on pitch, 11 per team, 23 identities, unique jerseys)

All verified exactly true of this half's GT (22 detections *every* frame; pipeline's split is
45.9/54.1 vs true 50/50). Applied **post-hoc** to the saved 45-min predictions (minutes per
variant, not 37 h):

| variant | GS-HOTA | Δ |
|---|---|---|
| 11-per-team applied per **frame** | 16.350 | −1.74 |
| jersey uniqueness | 16.424 | −1.67 |
| both | 14.674 | −3.42 |
| **11-per-team → one team per tracklet** | **19.724** | **+1.63** |
| one-shot team per tracklet | 19.295 | +1.20 |
| identity cap (347→58) | 18.094 | +0.001 AssA |
| junk-tracklet filter (any min length) | 18.097–18.108 | ≤+0.014 |

The per-frame version *raised per-detection team accuracy 76.17%→82.86% and lost 1.74 points* —
**124,186 team flips inside tracklets**. Same prior, same information, **3.37-point swing on
placement alone**. The transferable claim: *a prior only helps if imposed at the level the metric
evaluates (the track), and per-detection attribute accuracy can move opposite to GS-HOTA.*

---

## 6. Performance & memory engineering (what made a 45-min half possible)

Each of these independently blocked full-length runs. Details + backups:
[gsr-tracklab-patches.md](gsr-tracklab-patches.md).

| fix | before → after | validation |
|---|---|---|
| `merge_dataframes` quadratic (TrackLab) | per-batch cost 2.03× growth → 1.02× | identical GS-HOTA |
| `OptimizationConsistency` O(N²) (values-not-unique-ids) | 2h22m silent at 5 min; 25–45 h projected at 45 min → **0.006 s** (18,920×) | bit-identical (worst diff 0.0) |
| GTA pairwise distance on GPU | 54.88 GB at 45 min vs 16 GB GPU → 0.28 GB (198×) | exact identity, ≤5.96e-08 |
| `forget_columns` at teardown → module end, **deleted in place** | drop() freed nothing (datapipe pins the frame): 0.65 GB live → 0.08 GB | measured both ways |
| **glibc arena OOM** (the 10-min killer) | 35.5 GB heap, 0.23 GB live frame; `malloc_trim` per 100 batches reclaimed **58.2 GB** over the run; PRTReID peak 41.7→16.3 GB | trim can only return free memory — the run completing is the proof |
| gk_role debug CSVs (default-on) | ~16 GB disk + ~45 min per half → off | config |
| two-stage resumable runs | 37 h monolith → 19.6 h + 17.2 h with a state file between | 45-min half |

Refuted along the way (so nobody re-chases them): dataloader-worker accumulation across modules
(12→12, PSS *falls* at boundaries); worker copy-on-write; `cv2_load_image` lru_cache (4.65 GB but
constant in length); glibc mmap-threshold ratchet (synthetic repro: flat); **tracemalloc itself
accounted for ~10 GB** and produced one false "found it" — measure with PSS/`malloc_trim`, never
tracemalloc, and never sum RSS over forked workers (43 GB RSS = 12.9 GB PSS). Operational leaks:
OOM-killed parents orphan 12 workers holding ~9 GB indefinitely; the in-run evaluator deadlock
(§1.2).

---

## 7. Open items at time of writing

1. `track_buffer` sweep 30/90/250 at 10 min — running (`scratchpad/logs/buffer_sweep.log`)
2. Tracker association ReID swap away from `clip_duke.pt` — untried, best-motivated remaining idea
3. Non-finite calibration coordinates at source
4. Four exact CPU reductions from the module scan (~3–4 h of a 25 h run)
5. Landing the +1.63 track-level team prior in the pipeline (currently post-hoc only)
6. 132831 offset has no independent confirmation; its calibration defect (two transposed keypoints)
   is a separate known issue

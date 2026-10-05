# GSR baseline — Methods & Results draft (scoped for the paper)

Source text for the paper agent to adapt. Scope agreed on 2026-08-21: **(1) jersey-region gates,
(2) the engineering required to run a full 45-minute half — presented as a contribution,
(3) team assignment.** Negative results and abandoned experiments are excluded here (full record
in [gsr-experiment-log.md](gsr-experiment-log.md) if a reviewer response ever needs them).
**No manuscript files touched.**

All numbers below were verified against the primary files this session; sources in
`results/gsr/`.

---

## Methods

### Baseline pipeline

We adopt the SoccerNet game-state-reconstruction baseline (TrackLab / sn-gamestate): RF-DETR
person detection, ViTPose pose estimation, PRTReID appearance features and role classification,
BoT-SORT tracking, pose-based jersey-region extraction with PARSeq recognition, keypoint-based
calibration (thin-plate spline to calibrated pixels, then a homography to pitch coordinates),
global tracklet association (GTA), and per-tracklet majority voting for role, team, and jersey
number. Evaluation uses the official GS-HOTA metric in pitch space (5 m Gaussian tolerance),
which partitions detections into classes by (role, team, jersey): a detection with any incorrect
attribute cannot match ground truth at all.

### Jersey-region gates

The baseline extracts a torso region from each detection using pose keypoints, then gates the
region before recognition by minimum area (`min_roi_area`) and minimum oriented-box aspect ratio
(`min_obb_aspect_ratio`). At our broadcast resolution the median player box is 25×55 px and the
median proposed torso region is ~12×11 px (132 px²), so the stock gate of 500 px² / 0.6 passes
only 6.9% of detections: the recognizer is starved of candidates rather than failing on the ones
it sees. Of 31,107 detections in a 30-second clip, only 3.0% received a region proposal and 1.6%
produced a reading at confidence ≥ 0.8, reaching 8 of 65 tracklets.

We tuned both gates on a *validation-split* match (117093), sweeping
`min_roi_area ∈ {100…500}` × `min_obb_aspect_ratio ∈ {0.3, 0.6}`, and selected 100 / 0.6
(GS-HOTA 13.572 → 20.709 on the tuning match). The selected values were then applied unchanged to
the test match.

### Team assignment

The baseline clusters *tracklet-mean* ReID embeddings with 2-means and labels sides by mean pitch
position. On long sequences tracklets fragment, so tracklet-mean clustering degrades; we instead
cluster **per-detection** embeddings and assign each tracklet the majority label of its
detections (goalkeepers, referees and non-persons are excluded from clustering, as in the
baseline). Team accuracy is measured on geometrically matched prediction–ground-truth pairs
(Hungarian matching in pitch space, ≤ 2 m).

### Scaling the pipeline to a full half

The reference implementation is implicitly clip-scale: on full-half input (67,625 frames,
1.39 M detections) it fails before finishing, for reasons that are invisible at clip length
because each grows superlinearly with sequence length. Enabling the first complete 45-minute
runs required four fixes, each individually blocking:

| bottleneck | behaviour at full-half scale | fix |
|---|---|---|
| per-batch result merging was O(N) per batch (quadratic total) | per-batch cost doubled within a clip; projected ~70 h/half | merge restricted to the batch's own rows/columns; verified score-identical |
| consistency module iterated the track-id column's *values* rather than unique ids (O(N²)) | silent for 2 h 22 m on a 5-minute clip; 25–45 h projected per half | single grouped pass; **18,920×** faster, bit-identical output |
| global tracklet association materialised full n₁×n₂ pairwise-similarity tensors on the GPU to compute a scalar mean | 54.9 GB of allocations at 45 min vs a 16 GB GPU | exact separable identity (mean pairwise cosine = 1 − dot of mean unit embeddings); 0.28 GB, equal to the original within float32 rounding |
| glibc retained freed per-thread arena memory across the pipeline's many stages | resident memory reached 61 GB (kernel OOM kill) while live data was < 1 GB | periodic `malloc_trim`; 58.2 GB returned to the OS over one run, peak resident 16.3 GB |

We additionally run each half in two resumable stages (detection+pose → saved state → remaining
modules), so a failure in the second stage does not repeat ~17 h of GPU work. With these fixes a
full half completes in ~37 h on one RTX 4060 Ti (16 GB) and 61 GB RAM.

*(Suggested framing: these fixes are a contribution of the benchmark itself — full-match
evaluation is not merely "longer clips", and we release the patches so the baseline is runnable
at the scale the dataset is designed for.)*

### Scoring long sequences

Two robustness details matter at full-half scale and are handled by our released scorer:
calibration produces non-finite pitch coordinates for a small fraction of detections near the
horizon (0.16% at 10 min, 0.435% at 45 min); these are unmatchable in pitch space and are dropped
with their count reported. (The stock evaluator aborts on the first such value.)

---

## Results

### Jersey-region gates

Gates tuned on the validation match transfer directly to the test match (stock team assignment,
identical footage prefixes):

| clip length | stock gates (500/0.6) | tuned gates (100/0.6) | Δ |
|---|---|---|---|
| 30 s | 37.155 | 40.352 | +3.20 |
| 1 min | 33.471 | **57.824** | **+24.35** |
| 2 min | 32.274 | 44.810 | +12.54 |

GS-HOTA (higher is better). The gain is largest where tracklets are long enough to accumulate
readings but the stock gate starved them; it is the largest single configuration effect we
measured, and it is a *recall* fix — the recognizer was adequate, the funnel was not.

### Team assignment

At 5 minutes, tracklet-mean clustering collapses to near-chance team accuracy (51.6% on matched
pairs; two classes), because fragmented tracklets yield unreliable mean embeddings.
Detection-level clustering restores it:

| | team accuracy (≤2 m matched pairs) | GS-HOTA (5 min) |
|---|---|---|
| tracklet-mean clustering (stock) | 51.6% | 19.484 |
| **detection-level clustering** | **86.6%** | **30.974** |

On the full 45-minute half, detection-level team assignment holds at **76.17%** on 36,665 matched
pairs (role assignment: 97.92%). Enforcing the game's structure — exactly eleven players per team,
committed once per tracklet — raises the full-half score from 18.094 to **19.724** with no
retraining.

### Full-half result

With the fixes of §Methods, the pipeline completes a full 45-minute half end to end for, to our
knowledge, the first time with this baseline: 67,625 frames, 1,388,671 detections,
**GS-HOTA 18.094** (26.036 with attribute matching disabled, i.e. geometry and association only;
DetA 51.2, LocA 84.7). Accuracy on nested prefixes of the same footage peaks at one minute
(47.589) and declines monotonically with length, which is the paper's case that full-match
evaluation is a distinct and necessary protocol rather than a longer clip
(`results/gsr/length_curve.png`, `results/gsr/length_sweep_128057_1st.csv`).

### Limitations (suggested sentences)

Results are from a single test match (both halves staged; one evaluated end to end), so absolute
numbers carry no variance estimate. Gate values were selected on a validation match and applied
unchanged. Team-assignment variants cross over with length (tracklet-mean is better below two
minutes); all tables above use one fixed configuration. The structural-prior improvement (+1.63)
is applied post hoc to saved predictions and has not been validated by an end-to-end rerun.

---

## Numbers checklist for the paper agent

- 57.824 / +24.35 (jersey gates, 1 min) — `results/gsr/length_sweep_128057_1st_tracklet_team.csv`, `jersey_sweep.csv`
- 86.6% / 30.974 (team, 5 min) — `config5.csv`, structural_priors write-up
- 76.17% team, 97.92% role at 45 min — `results/gsr/attribute_ablation_45min.md`
- 18.094 / 26.036 / 19.724 (full half) — `results/gsr/score_45min_128057_1st.json`, `score_45min_prior_team_vote.json`
- 18,920× / 54.9→0.28 GB / 58.2 GB trimmed — `docs/gsr-tracklab-patches.md` (commit messages carry the measurements)
- funnel numbers (3.0% / 1.6% / 12×11 px / 6.9%) — report §jersey, `experiment_inventory.json`

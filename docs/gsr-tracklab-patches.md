# Patches to TrackLab and sn-gamestate

The GSR baseline does not run on stock TrackLab. It needs the changes below, which live in two
checkouts **outside this repository**:

| tree | path | upstream commit patched |
|---|---|---|
| TrackLab | `/home/atom/soccernet/tracklab` | `1b48e40c` |
| sn-gamestate | `/home/atom/soccernet/gsr` | `c7beb02` |

Both are git checkouts, so `git -C <tree> diff HEAD` shows the current delta. This file explains
**why** each change exists and what it measurably bought, because a diff alone does not say which
changes were validated and which were tried and rejected.

Every modified file has a sibling `*.backup-<reason>` holding the pre-change version, so any single
patch can be reverted and re-measured in isolation.

---

## 1. Correctness — required for the pipeline to run at all

### `gsr/sn_gamestate/gta/connect_track.py` — remove reads of attributes that do not exist
`use_spatial_connect=True` crashed with `AttributeError: 'Tracklet' object has no attribute 'role'`.
Two functions read `role` and `jersey_number` off `Tracklet`, which never carries them. The reads
were dead — nothing downstream consumed the results — so they were deleted rather than populated.
A stray debug `print` in the same file was removed with them.

Backup: `connect_track.py.backup-pre-role-fix`.

> Still open: this module's spatial thresholds are in **pixels** but are applied to `bbox_pitch`,
> which is in **metres**. It is enabled and it runs, but the thresholds are not meaningful. Treat
> any GTA spatial-connect result as unvalidated until they are converted.

---

## 2. Performance — needed for sequences longer than a few minutes

### `tracklab/tracklab/engine/engine.py` — de-quadratify `merge_dataframes`
Per-batch results were merged with a pattern that rebuilt the whole detections frame each time, so
cost grew with the square of sequence length. This is the main reason 45-minute halves were
projected at ~63 h. Fixed in place; verified to produce **identical GS-HOTA** before and after.

Backup: `engine.py.backup-pre-quadratic-fix`.

### `tracklab/tracklab/engine/offline.py` — apply `forget_columns` at module end
Modules declare `forget_columns` to release heavy intermediates, but `TrackerState` only applied it
during **video teardown** — after the whole video finished, long past the point where the memory
mattered. `PRTReID` declares `["embeddings", "body_masks", "all_role_confidences"]`; `body_masks` is
a 1×64×32 float32 array, **8 KB per detection**, that nothing downstream reads. It was therefore
carried through all eleven remaining modules: **2.5 GB at 10 minutes, 11 GB at 45 minutes.**

`_drop_forgotten_columns` releases a finished module's forget-columns, but **only those no later
module declares as an input**. That exclusion is load-bearing: `embeddings` is in PRTReID's forget
list, yet `main_subject_filter` and `team` both consume it, so an unconditional drop would break the
run.

Backup: `offline.py.backup-pre-forget-fix`.

> **Scope, stated honestly:** this does *not* explain the 10-minute OOM. That kill happened at 61%
> *through* PRTReID, and this fix only takes effect once PRTReID has finished, so it was never
> reached. It is a real defect worth fixing for everything downstream and for 45-minute runs; it is
> not the cause of that crash. See §4.

---

## 3. New modules (added, not patched)

| module | config | status |
|---|---|---|
| `calibration/manual_calib_distorted.py` | `manual_calib_distorted.yaml` | **in use** — TPS (distorted px → calibrated px) then `inv(H_pc)` to pitch metres |
| `team/tracklet_team_detection_level.py` | `team/detection_level.yaml` | **conditional** — see below |
| `pose_detector/vitpose_strided.py` | `pose_detector/vitpose_strided.yaml` | **not in use** — sub-2× speedup cost too much accuracy |
| `calibration/manual_calib_metre.py` | `calibration/manual_calib_metre.yaml` | **rejected, kept as a negative result** |

### `tracklet_team_detection_level` — clusters detections, not tracklets
Stock team assignment k-means over *tracklet-mean* embeddings collapses when tracklets fragment. At
5 minutes team accuracy was 51.6% — indistinguishable from chance on two classes. Clustering at
detection level and voting per tracklet lifts it to 86.6%, and GS-HOTA 19.484 → 30.974.

**It is not a free win.** It is *worse* on short sequences and better on long ones:

| length | tracklet-level | detection-level |
|---|---|---|
| 30 s | **40.352** | 37.155 |
| 1 min | **57.824** | 47.589 |
| 2 min | — | 39.882 |
| 5 min | 19.484 | **30.974** |

There is no single best team module across lengths. Any length-sweep must fix one config for all
points or it measures the config change, not the length.

### `manual_calib_metre` — rejected
Fits the homography objective in metre space rather than pixel space. Measured **−2.9 and −8.1**
GS-HOTA. The default was reverted to `homography_objective: pixel`. Kept so the experiment is not
repeated.

---

## 4. Known memory behaviour

- **Orphaned dataloader workers.** When the parent is OOM-killed, its `pt_data_worker` children are
  reparented to init and **never reaped**. Twelve such workers from a killed 10-minute run were
  found alive 23 hours later holding **8.79 GB PSS**. Check for and kill these before any long run:

  ```bash
  pgrep -f 'sn_gamestate.main' | while read p; do [ "$(ps -o ppid= -p $p | tr -d ' ')" = 1 ] && echo $p; done
  ```

- **Measure PSS, not RSS.** Forked workers share most of their pages with the parent, so summing RSS
  counts the same memory twelve times. A tree that reads as 43 GB of RSS was 12.9 GB of PSS. An
  early claim that "memory is not a constraint" came from parent-only RSS and was wrong.

- `num_cores=12` multiplies whatever the parent holds by the fraction each worker dirties, and the
  parent's state grows linearly with sequence length. Length and worker count are not independent
  knobs.

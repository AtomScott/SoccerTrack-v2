# Running the GSR baseline on SoccerTrack v2

The GSR baseline is **TrackLab / sn-gamestate**, run as a single command. Do not write a
custom detector+tracker script: the released GSR annotations are *already* SoccerNet
GameState, so the dataset is natively what TrackLab consumes. What this directory provides is
the **staging** needed to get a 45-minute half in front of it, plus the harness for the
sequence-length study.

External pipeline tree (not part of this repo): `/home/atom/soccernet/gsr`, env `gsr/.venv`
(Python 3.9), config `sn_gamestate/configs/soccertrack.yaml`.

## Why staging is needed at all

TrackLab reads frames from `<sequence>/img1/000001.jpg`, never from video, and it takes its
frame index from the ground-truth `images` block. Three things in the release get in the way:

| problem | consequence | handled by |
| --- | --- | --- |
| frames only exist as `.mp4` | TrackLab cannot read the sequence | `stage_soccernetgs.py` extracts `img1/` |
| `images[].width/height` say `3840x1504`; the real frames are `4096x1080` | stale metadata reaches downstream modules | corrected during staging |
| `info.id` is `"1"` in **all 20** released files, and `info.name` is `CLPD-<match>` for **both** halves | sequence ids and output filenames collide | unique id + name assigned per half |
| the GT annotates ~25 more frames than the video contains | TrackLab requests JPEGs that cannot exist | `seq_length` clamped to the real frame count |

See `docs/format-gsr.md` for the full format description and defect list.

## Reproduction sequence

**1. Stage a half** (~110 GB of JPEGs per half at `-q:v 2`, which matches the 1.67 MB/frame of
the pre-existing staged clip; use `/mnt/storage`, not `/data`):

```bash
python scripts/gsr/stage_soccernetgs.py --match 128057 --half 1st --split test \
  --out-root /mnt/storage/SoccerTrack-v2/SoccerNetGS
```

**2. Generate calibrated keypoints.** The release ships `<match>_calibrated_keypoints.json`
for **117093 only**, and `manual_calib_distorted` needs both a distorted and a calibrated set,
so every other match needs this:

```bash
python scripts/gsr/make_calibrated_keypoints.py --matches 128057 132831
```

The calibrated set is only an *intermediate* space — TPS maps distorted→calibrated and `H_pc`
maps pitch→calibrated, both fit from the same points, so the canvas cancels out. The script
proves this by regenerating 117093's set and comparing the full image→pitch chain against the
shipped one (agreement: **0.0000 m** over 725 on-pitch grid points). It still normalises the
canvas, because a raw `balance=0` undistortion puts 128057's control points at
x ∈ [−7973, 11234] for a 4096-wide image, which makes the TPS extrapolate wildly.

For **132831** it picks up the corrected keypoints from `data_corrections/` automatically —
the two transposed points that wrecked that match's calibration.

**3. Run the pipeline:**

```bash
cd /home/atom/soccernet/gsr && .venv/bin/python -u -m sn_gamestate.main -cn soccertrack \
  experiment_name=soccertrack-128057-1st \
  'dataset.vids_dict.test=[CLPD-128057-1st]' \
  dataset.nvid=-1 \
  dataset.dataset_path=/mnt/storage/SoccerTrack-v2/SoccerNetGS \
  modules.calibration.distorted_keypoints_json=/data/share/SoccerTrack-v2/data/raw/128057/128057_keypoints.json \
  modules.calibration.calibrated_keypoints_json=/data/share/SoccerTrack-v2/data/raw/128057/128057_calibrated_keypoints.json \
  visualization.cfg.save_videos=False use_rich=False num_cores=12
```

Two flags are not optional and are easy to get wrong:

- **`use_rich=False`** — with rich enabled the progress display detects a non-TTY and writes
  *nothing* to a redirected log. You are blind for the whole run.
- **`num_cores=12`** — this feeds `engine.num_workers` for every module's DataLoader
  (`configs/engine/offline.yaml`). The shipped value of `1` starves the GPU: measured 2.05x
  speedup (2.14 → 1.04 s/batch, GPU idle samples 11/18 → 3/24) on a 24-core machine.

## Cost, measured

On one RTX 4060 Ti (16 GB) at 4096x1080, for a 750-frame clip, whole 14-module pipeline:

| module | batches | time |
| --- | --- | --- |
| ViTPose | 512 | **10:31** |
| RF-DETR | 94 | 1:36 |
| PRTReID | 256 | 0:51 |
| PARSeq | 128 | 0:27 |
| PoseBasedRegionExtractor | 512 | 0:09 |
| ManualCalib | 750 | 0:05 |
| **total** | | **~15:00** |

ViTPose is 56% of it, at ~26 crops/s. Scaling to a 67,625-frame half gives roughly **20 h**.

**A quadratic-time defect in TrackLab had to be fixed to get there.** `merge_dataframes`
(`tracklab/engine/engine.py`) is called once per batch and was O(len(accumulated frame)) twice
over, so per-batch cost grew linearly and total cost quadratically: measured 1.02 s/batch at
batch 0 rising to 3.74 s by batch 2353. Patched, the same run holds 0.90–0.97 s/batch flat.
That fix lives in the external tracklab tree, not here; the original is preserved as
`engine.py.backup-pre-quadratic-fix`. It was validated by running a full 750-frame pipeline
before and after: GS-HOTA 29.135% both times, metric summary files byte-identical, and the
only prediction differences were float noise at 3.9e-11 m.

## Evaluation

Pitch-space GS-HOTA only. There are **no ground-truth detections** for this dataset — the
`bbox_image` values are auto-generated and are not to be evaluated against. Attributes matter
enormously; on the 30-second clip:

| configuration | GS-HOTA | DetA | AssA | LocA | DetRe |
| --- | --- | --- | --- | --- | --- |
| roles+teams+jersey (official) | 29.13 | 9.70 | 87.60 | 92.15 | 17.57 |
| roles+teams, no jersey | 49.99 | 34.99 | 71.67 | 86.61 | 50.65 |
| roles only | 62.53 | 55.87 | 70.34 | 87.14 | 69.55 |
| no attributes (geometry only) | 64.41 | 59.61 | 70.09 | 86.50 | 72.14 |

Detection and pitch projection are good: **72% recall**. Jersey numbers cost **20.9 points**
and team classification **12.5**. Do not quote AssA 87.60 as an association result — with
attributes on, only detections matching on every attribute can pair at all, so that figure is
a selection effect.

## Sequence-length study

Whether accuracy degrades with sequence length is testable, and the mechanism is specific:
the detector, pose, ReID and BoT-SORT are all causal and cannot be affected by later frames,
but `GTALink`, `MajorityVoteTracklet`, `TrackletTeamClustering` and `TrackletTeamSideLabeling`
are global over the whole video. More minutes means more tracklets.

```bash
# nested prefixes of one half -- costs ZERO extra disk, img1 is a symlink in each
python scripts/gsr/stage_prefix_sequences.py --source CLPD-128057-1st

# after running the pipeline once per prefix
python scripts/gsr/score_length_sweep.py --out sweep_scores.json
python scripts/gsr/plot_length_sweep.py --work <dir with metrics/>
```

`score_length_sweep.py` scores every run twice: on its own full length ("what do you get on
N minutes") and on the **same first 750 frames** shared by all runs, which isolates length
from content because the footage compared is identical.

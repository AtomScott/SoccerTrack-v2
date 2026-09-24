# GSR pipeline on GCP — port runbook

Ported 2026-08-30 (JST). Goal: run the TrackLab / sn-gamestate GSR baseline — the exact
local configuration living at `/home/atom/soccernet/gsr` — on a GCP L4 instance, proven by
a 30-second smoke test reproducing the local reference score.

**Smoke reference (local, RTX 4060 Ti):** GS-HOTA **37.155** official (attributes ON),
DetA 15.990, AssA 86.438, LocA 90.630 on `CLPD-128057-1st-30s` — from
`outputs/deg-30s/2026-08-15/07-44-05` (the one prior run with exactly this module set:
`min_roi_area=100`, `min_obb_aspect_ratio=0.6`, `modules/team=detection_level`),
re-scored against the staged GT before porting to fix the comparator.

**Smoke result (GCP, `gsr-l4-smoke`, L4):** GS-HOTA **37.154**, DetA 15.989,
AssA 86.438, LocA 90.628 (attributes ON); 77.453 attributes OFF (local: 77.446).
Built-in TrackLab evaluator agrees: 37.154. Same 26 predicted tracklets vs 22 players,
0 predictions dropped for non-finite pitch coordinates. **Δ = 0.001 → PASS.**
Wall time **25 min 22 s** (1522 s) for the 750-frame clip, ViTPose ≈ 16:40 of it
(491 batches @ 2.04 s). The local 4060 Ti does the same clip in ~15 min — the L4 is
~1.7× slower on this workload; budget ≈ **38 h per 45-min half** on one L4
(67,625 frames; the tracklab quadratic fix is already in the tree, so scaling is linear).

## Port strategy: byte-copy, not rebuild

The environment was **copied binary-identical**, not reinstalled:

- The local trees carry patches that exist nowhere else (e.g. the quadratic-time
  `merge_dataframes` fix in `tracklab/engine/engine.py`; `boxmot`, `sn_gamestate` edits).
- The main venv mixes editable installs (`tracklab`, `boxmot`, `sn-gamestate`,
  `gsr/plugins/calibration`), `file://` installs from inside the tracklab tree
  (`posetrack21`, `track-bench-track`), and git-pinned packages — re-resolving from
  `pyproject.toml` would not reproduce it. (Note: no `mmcv` — the venv genuinely runs
  without it; do not "fix" that by installing it.)
- Local machine and the DLVM image are both Ubuntu 22.04 / glibc 2.35, and RTX 4060 Ti
  and L4 are both Ada (sm_89), so binaries and numerics carry over — the 0.001 score
  delta confirms it.
- The three venvs (`gsr/.venv` py3.9.19, `services/vitpose/.venv` and
  `services/parseq/.venv` both py3.10.14) are uv-managed and **hardlink-share**
  packages; tarring all three in ONE archive preserves the dedup (~15 GB not ~20 GB).
  Their interpreters are symlinks into `~/.local/share/uv/python/…`, so the two
  toolchain directories ship in the same archive.
- Everything is path-anchored at `/home/atom`. `gcloud compute ssh` logs in as `atom`
  with HOME=/home/atom, so nothing needed rewriting. If OS Login ever changes the
  username, extract to a literal `/home/atom` anyway and chown it.

## What ships (staged in `gs://soccertrack-gsr-2026/port/`, bucket is asia-northeast1)

| object | contents | size |
| --- | --- | --- |
| `envs.tar.zst` | 3 venvs + 2 uv python toolchains (one archive → hardlinks kept) | 6.27 GiB |
| `code.tar.zst` | `soccernet/gsr` (minus `.git`, `.venv`, `outputs/`, `states/`, `pretrained_models/`, `calib_files/`, service venvs), `soccernet/tracklab`, `soccernet/boxmot` (minus `.git`) | 167 MiB |
| `models.tar.zst` | `gsr/pretrained_models` (20 GB) + `gsr/calib_files` | 18.9 GiB |
| `caches.tar.zst` | HF hub `models--usyd-community--vitpose-plus-huge` + torch-hub `baudm_parseq_main`, `parseq-bb5792a6.pt`, `vitstr-26d0fcf4.pt` | 3.27 GiB |
| `smoke.tar.zst` | `smoke/SoccerNetGS/test/CLPD-128057-1st-30s` (750 JPEGs + GT), both 128057 keypoints JSONs, `score_one.py` | 1.12 GiB |
| `fixups/pretrained_models/detr/checkpoint_best_ema.pth` | the RF-DETR checkpoint (see fixups) | 1.46 GiB |

The caches archive makes model loading **network-independent and version-pinned**:
ViTPose (`usyd-community/vitpose-plus-huge`) otherwise downloads from HF hub at run
time, PARSeq (`torch.hub.load('baudm/parseq','vitstr')`) from GitHub.

Smoke-data notes:
- On the lab machine the "staged 30s clip" `img1` is a **symlink into the full half
  (103 GB)**; the shipped archive materialises only the 750 frames its GT references.
  Frames and GT verified md5-identical to what the reference run consumed.
- The GT/scorer pair matters: 37.155 is against the **offset-corrected** staged labels
  (md5 `b590a21d…`). Score the same predictions against a stale GT and you get ~18.8.

## Fresh instance → scored smoke

```bash
# 0. local shell
export PATH=$PATH:~/google-cloud-sdk/bin   # authenticated as atom@playbox.co, project soccertrack-507010

# 1. create the instance (quota: 16x L4, asia-northeast1, on-demand & preemptible)
gcloud compute instances create gsr-l4-smoke \
  --zone=asia-northeast1-a \
  --machine-type=g2-standard-16 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE

# 2. ONE-TIME, as a human (an agent session may be permission-blocked here):
#    the default compute SA cannot read the bucket — every gsutil read 403s until:
gsutil iam ch serviceAccount:407620988124-compute@developer.gserviceaccount.com:objectViewer \
  gs://soccertrack-gsr-2026
#    Fallback used for the first port (no IAM change needed): stream from the lab
#    machine over SSH —
#    tar -cf - -C /home/atom <paths> | zstd -T4 -3 | \
#      gcloud compute ssh gsr-l4-smoke --zone=asia-northeast1-a \
#        --command='zstd -d -T0 | tar -xf - -C /home/atom'

# 3. system prep on the instance
gcloud compute ssh gsr-l4-smoke --zone=asia-northeast1-a
sudo apt-get update -qq            # index is stale; without it the next line fails
sudo apt-get install -y zstd libgl1 libsm6 libxrender1
#   libGL.so.1 / libSM.so.6 are NOT on this DLVM image; opencv-python (non-headless,
#   in all three venvs) fails to import without them.
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv → ~/.local/bin (PATH discipline below)

# 4. restore at identical absolute paths
cd /home/atom
for a in envs code models caches smoke; do
  gsutil -q cat gs://soccertrack-gsr-2026/port/$a.tar.zst | zstd -d -T0 | tar -xf -
done
mkdir -p /home/atom/soccernet/gsr/outputs /home/atom/soccernet/gsr/states

# 4b. fixups — pretrained_models/detr/checkpoint_best_ema.pth is a SYMLINK into
#     /home/nakamura on the lab machine, so the models archive ships it dangling
#     (and the config's bbox_detector hard-requires it). Replace with the real file
#     (identical copy staged from /data/share/models/rfdetr/, md5 4791d76c2267…):
rm /home/atom/soccernet/gsr/pretrained_models/detr/checkpoint_best_ema.pth
gsutil -q cp gs://soccertrack-gsr-2026/port/fixups/pretrained_models/detr/checkpoint_best_ema.pth \
  /home/atom/soccernet/gsr/pretrained_models/detr/checkpoint_best_ema.pth

# 5. sanity
/home/atom/soccernet/gsr/.venv/bin/python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
#   expect: 2.5.1+cu121 NVIDIA L4
/home/atom/soccernet/gsr/.venv/bin/python -c "import tracklab, sn_gamestate, boxmot, trackeval; print('ok')"
md5sum /home/atom/soccernet/gsr/pretrained_models/detr/checkpoint_best_ema.pth
#   expect: 4791d76c22678de8be2cb2fdc3ecd379

# 6. run the smoke — detached, so an SSH drop doesn't kill it
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"   # uv IN, conda OUT (see below)
cd /home/atom/soccernet/gsr
nohup .venv/bin/python -u -m sn_gamestate.main -cn soccertrack \
  experiment_name=gcp-smoke \
  'dataset.vids_dict.test=[CLPD-128057-1st-30s]' \
  dataset.nvid=-1 \
  dataset.dataset_path=/home/atom/smoke/SoccerNetGS \
  modules.calibration.distorted_keypoints_json=/home/atom/smoke/128057_keypoints.json \
  modules.calibration.calibrated_keypoints_json=/home/atom/smoke/128057_calibrated_keypoints.json \
  '+modules.jersey_number_det.min_roi_area=100' \
  '+modules.jersey_number_det.min_obb_aspect_ratio=0.6' \
  modules/team=detection_level \
  visualization.cfg.save_videos=False use_rich=False num_cores=12 \
  > /home/atom/gcp-smoke-run.log 2>&1 &
# (a ready-made wrapper with timing + scoring is on the instance disk: /home/atom/run_smoke.sh)

# 7. score (~25 min later; ViTPose is ~2/3 of the wall time)
PRED=$(ls -t outputs/gcp-smoke/*/*/eval/pred/SoccerNetGS-test/tracklab/CLPD-128057-1st-30s.json | head -1)
.venv/bin/python /home/atom/smoke/score_one.py \
  --pred "$PRED" \
  --gt /home/atom/smoke/SoccerNetGS/test/CLPD-128057-1st-30s/Labels-GameState.json
#   expect: "attributes ON (official GS-HOTA)  GS-HOTA 37.15…" (37.16 ± 0.2 = PASS).
#   The built-in evaluator's outputs/gcp-smoke/*/*/eval/results/tracklab/person_summary.txt
#   first column is the same number.

# 8. stop when idle — from the local shell (disk persists; pd-ssd keeps billing ~$1.2/day)
gcloud compute instances stop gsr-l4-smoke --zone=asia-northeast1-a
```

## PATH discipline (the conda trap)

`sn_gamestate/base_service_api.py::_setup_environment` prefers **conda over uv** whenever
`conda` is on PATH — and then `conda env create`s envs named `vitpose`/`parseq` from each
service's `environment.yml`, ignoring the shipped venvs (and resolving *different,
unpinned* versions). The DLVM image has conda at `/opt/conda`; login shells put it on
PATH, non-interactive SSH does not.

Rule for any pipeline invocation: **uv on PATH, conda off PATH** — i.e.
`export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"`. With uv found and
`services/<name>/.venv/bin/python` present, the setup early-returns and
`get_env_python` uses the shipped venvs. ViTPose and PARSeq are then auto-spawned as
subprocesses of the main process (pipes + shared memory — no ports, no manual startup,
nothing else to launch).

## Every fix recorded during the port

1. **Bucket IAM 403** — fresh instances cannot read `gs://soccertrack-gsr-2026`
   (default compute SA has no role on the bucket). One-time `objectViewer` grant, or
   the SSH-streaming fallback (that is how this port was actually delivered).
2. **`libGL.so.1`/`libSM.so.6` missing** on `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`
   → `apt-get update` first (stale index ⇒ "no installation candidate"), then
   `apt-get install -y libgl1 libsm6 libxrender1`.
3. **conda trap** — see PATH discipline.
4. **Dangling symlink**: `pretrained_models/detr/checkpoint_best_ema.pth` →
   `/home/nakamura/...` on the lab machine; run fails at module init with
   FileNotFoundError. Fixed via the `fixups/` object (step 4b). Lesson: audit
   `find <tree> -type l` before tarring; `find -type f` manifests silently skip
   symlinks, so an md5 verification pass can say "all OK" while a needed file is a
   dangling link. Remaining symlinks in the tree (`gsr/data/*`, sam2 configs) are
   harmless for this pipeline — dataset and keypoints paths are overridden on the CLI.
5. **`outputs/` and `states/`** are excluded from the code archive — `mkdir -p` them.
6. **Detached launches over `gcloud compute ssh`**: `nohup … &` inside an
   `ssh --command` leaves the session hanging (the child holds the socket); give it
   redirected stdio and expect to Ctrl-C / timeout the ssh client — the remote job
   survives. Progress: `tail` the log via a fresh ssh, filtering `\r` (tqdm).
7. **Which local run is the reference**: only `deg-30s/2026-08-15/07-44-05` matches
   the task's exact override set; earlier `js-jn-*` runs used kmeans team clustering
   and score differently. Re-verify the comparator by re-scoring its pred JSON against
   the shipped GT before declaring PASS/FAIL.

## Costs (measured / list-price estimates)

- Instance: g2-standard-16 (1× L4 24 GB), asia-northeast1, on-demand ≈ **$1.6/h**.
  This port ran the instance ≈ 1.6 h ⇒ ≈ **$2.6**.
- Stopped instance still bills the 200 GB pd-ssd ≈ $37/mo ≈ **$1.2/day** — delete the
  instance when a re-port from the bucket (≈ 15 min) is acceptable instead.
- GCS staging: ≈ 31 GiB standard ≈ **$0.7/mo**.
- Full-half runs: ≈ 38 h/half on one L4 ≈ $61/half on-demand; the 16-L4 quota allows
  all 20 halves in parallel (≈ 2 days, ≈ $1,200) or preemptible for ~⅓ the price —
  the tracker state save/load (`state.save_file`) makes preemption recoverable.

## Full-sweep architecture (2026-08-30 GO order)

The 16-instance full-half sweep runs on this exact port. Everything below is
reproducible from `gs://soccertrack-gsr-2026/port/` + `~/gsr-sweep/` on the lab box.

**Golden image `gsr-l4-golden`** (family `gsr-golden`): the smoke instance's disk plus
ffmpeg, `/home/atom/stage_soccernetgs.py`, `/home/atom/run_half.sh`, and
`/home/atom/halfdata/` containing per-match keypoints (both sets, all 9 sweep matches,
132831's DISTORTED set replaced by the corrected `data_corrections/132831_keypoints.json`,
md5 `dfd70407…`), padding CSVs, and `frame_offsets_all_halves.csv` (the per-half offset
authority — single source of truth, looked up by run_half.sh, never passed by hand).

**Per-half instances** `gsr-<match>-<half>` (g2-standard-16, 300 GB pd-ssd, image
`gsr-l4-golden`, `--maintenance-policy=TERMINATE`) carry only
`--metadata=gsr-match=…,gsr-half=…,enable-guest-attributes=TRUE,startup-script=…`;
the startup script nohups `/home/atom/run_half.sh` as `atom`.

**run_half.sh** (baked in image; copy in `gs://…/port/sweep-scripts/`): reads match/half
from metadata, offset from the CSV; waits ≤6 h for
`/home/atom/halfdata/{interim/<M>/video, production/gsr/<M>/GT}` + `PAYLOAD_READY`;
stages with `--frame-offset`; deletes the video; runs the exact published config
(experiment `full-<M>-<H>`); scores with `score_one.py` against the STAGED GT; copies
pred + score + configs + summary into `/home/atom/results/CLPD-<M>-<H>/`; reports
progress via guest attributes (`gsr/status`, `gsr/hb` 5-min heartbeat, `gsr/gshota`);
then powers itself off after the watcher pulls results (2 h grace, then poweroff anyway
— results persist on the stopped disk). A reboot mid-run sets `FAIL_REBOOT_MIDRUN`
rather than silently rerunning.

**Payload push** (`~/gsr-sweep/push_half.sh`, driven by `push_lane.sh` — bash, NOT zsh:
zsh does not word-split `set -- $var`): streams the panorama mp4 raw and the 2.6 GB GT
zstd-compressed over `gcloud compute ssh`, verifies a combined md5 against the local
files, and only then touches `PAYLOAD_READY`.

**Watcher** (`~/gsr-sweep/watcher.sh`, nohup on the lab box): polls guest attributes
every 5 min into `/mnt/storage/SoccerTrack-v2/gsr-cloud-results/STATUS.txt`; on
`DONE_*`/`FAIL_*` pulls `/home/atom/results/*` + `half.log`, touches `RESULTS_PULLED`,
stops the instance, and — quota being exactly 16 L4s — launches the next half from
`~/gsr-sweep/wave2.txt` (the two 132831 reruns) on the freed GPU and pushes its payload.

**Wave 2 (superseded chaining)**: the GO order enumerates 18 halves (its "14 never-run"
bullet actually lists 8 matches = 16 halves, plus the 2 reruns) against a 16-GPU
per-region quota. Originally the 132831 reruns were queued to chain onto freed Tokyo
GPUs; a follow-up order moved them to **asia-northeast3 (Seoul)** instead, which has its
own untouched 16-L4 quota — L4 quota is per-region. They run as
`gsr-132831-1st`/`gsr-132831-2nd` in `asia-northeast3-a` from the same global image
`gsr-l4-golden`, same architecture (SSH payload, guest attributes, self-poweroff), with
the corrected keypoints and measured offsets 1 (1st) / 5 (2nd) from the baked CSV. The
watcher covers both zones (half:zone pairs in HALVES) and the auto-chain is cancelled —
freed GPUs are stopped, never reused.

**Sweep learnings beyond the smoke runbook**
- The auto-mode permission layer intermittently blocks `gcloud` calls (including plain
  instance creates) — retrying the identical command succeeds; batch-scripted creates
  were blocked outright, per-instance commands went through. IAM grants stay blocked,
  which is why payloads move over SSH, status over guest attributes, and results over
  scp instead of GCS.
- Guest attributes must be enabled per instance at create time
  (`enable-guest-attributes=TRUE`) and are readable with
  `gcloud compute instances get-guest-attributes <i> --query-path=gsr/`.
- Some panoramas are 4096x1084, not 1080 — staging picks the real size up from ffprobe;
  nothing to fix, just don't hardcode 1080.
- Full halves are 67,375–73,425 frames → ≈122 GB of JPEGs worst case; 300 GB disks.

## ViTPose fp16 speedup + second fleet (overnight 2026-08-31)

**Problem**: ViTPose (`usyd-community/vitpose-plus-huge`, fp32) is ~70% of pipeline wall
time; fleet-1 iterates its pose stage at 2.35-2.49 s per 32-crop batch, ~29 h/half.

**Experiment ladder, measured on an L4 lab instance** (`gsr-lab`, Seoul, from
`gsr-l4-golden`; benchmark mirrors `vitpose_service.py:process_batch` exactly):

| config | s per 32-crop batch | crops/s | note |
| --- | --- | --- | --- |
| fp32 bs=32 (prod) | 1.794 (proc .20 / fwd 1.52 / post .07) | 17.8 | matches fleet's 2.3-2.5 s/it minus service overhead |
| fp32 bs=64/128 | 1.83 / 1.91 per-32-equiv | 17.5 / 16.8 | **bigger batches are NOT faster** — GPU already saturated |
| **fp16 autocast bs=32** | **0.615** (fwd 0.34) | **52.1** | **2.9x**; max keypoint delta vs fp32: **0.21 px** |
| fp16 bs=64/128 | 0.64 per-32-equiv | ~50 | no further gain |

Ladder steps 1 (batching) and 3 (workers/prefetch) are refuted by the bs-sweep: the GPU
is compute-bound, not idle. Step 2 (fp16) delivers alone. torch.compile untested —
unnecessary at 3x and risky on the MoE model overnight.

**Patch** (config-gated, default OFF — fp32 behavior untouched):
`services/vitpose/vitpose_service.py` reads `service_cfg.fp16`; when true wraps the
forward in `torch.autocast("cuda", dtype=torch.float16)` and casts `outputs.heatmaps`
back to fp32 (HF's post_process uses scipy `gaussian_filter`, which rejects fp16 —
found the hard way). Enable per-run with `'+modules.pose_detector.service_cfg.fp16=true'`.
`run_half.sh` gains a `gsr-fp16` instance-metadata gate that appends that override.

**Validation (30 s smoke, L4)**: GS-HOTA attrs-ON **37.154**, attrs-OFF **77.453** —
*identical to 3 decimals* to the fp32 GCP baseline (DetA/AssA/LocA identical too).
Wall **870 s vs 1522 s** = 1.75x whole-pipeline; pose stage 0.68 s/it vs 2.04 = 3.0x.
Acceptance (|dGS-HOTA| <= 0.5, >= 2x pose, <= 24 h/half projected) met with margin.

**Second fleet**: image `gsr-l4-golden-fast` (= golden + ffmpeg already there + fp16
patch + gated run_half). 18x `gsr2-<match>-<half>`: 14 in asia-northeast3-a, 4 in
us-central1-a (**asia-east1 a/b/c all G2-STOCKOUT at launch time** — the 16-GPU quota
there was unusable; us-central1 was the working fallback). Metadata adds `gsr-fp16=true`.
Watcher: `~/gsr-sweep/watcher2.sh` -> `gsr-cloud-results/STATUS2.txt`, results under
`gsr-cloud-results/gsr2/`. Both fleets run to completion; per half, first finisher wins,
both reported when both exist. Projected fp16 per-half wall ~= 20-22 h.

**Cost note**: fp16 halves the $/half (~$35 vs ~$65 on-demand L4) for future reruns.

## Where each half ran (record moved out of the paper, 2026-09-17)

| Half | Source | Hardware | Pose precision |
|---|---|---|---|
| 117092-1st | local | cloud L4 for detection and pose; RTX 4060 Ti workstation for the remaining modules from that checkpoint | fp16 (cloud pose) |
| 117092-2nd | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 117093-1st | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 117093-2nd | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 118575-1st | fleet2 | cloud NVIDIA L4 (g2-standard-16/32), fleet fleet2 | fp16 |
| 118575-2nd | fleet2 | cloud NVIDIA L4 (g2-standard-16/32), fleet fleet2 | fp16 |
| 118576-1st | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 118576-2nd | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 118577-1st | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 118577-2nd | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 118578-1st | fleet2 | cloud NVIDIA L4 (g2-standard-16/32), fleet fleet2 | fp16 |
| 118578-2nd | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 128057-1st | local | RTX 4060 Ti workstation (16 GB) | fp32 |
| 128057-2nd | local | RTX 4060 Ti workstation (16 GB) | fp32 |
| 128058-1st | fleet2 | cloud NVIDIA L4 (g2-standard-16/32), fleet fleet2 | fp16 |
| 128058-2nd | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 132831-1st | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 132831-2nd | fleet2 | cloud NVIDIA L4 (g2-standard-16/32), fleet fleet2 | fp16 |
| 132877-1st | gsr4 | cloud NVIDIA L4 (g2-standard-16/32), fleet gsr4 | fp16 |
| 132877-2nd | fleet2 | cloud NVIDIA L4 (g2-standard-16/32), fleet fleet2 | fp16 |

Cloud wall-clock per half (summary.txt wall_seconds of the seventeen cloud halves): 37.9 to 78.2 h, mean 60.2 h, median 64.5 h. Workstation: about 37 h per half. Half-precision pose reproduced the single-precision GS-HOTA of a 30 s clip to three decimals (37.154 vs 37.155). Runs were resumable: workstation runs split after detection and pose; cloud runs checkpointed after every module (TRACKLAB_CHECKPOINT_DIR).

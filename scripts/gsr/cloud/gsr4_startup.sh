#!/bin/bash
# gsr4 startup v3 (2026-09-11). Runs as root at every boot. Order matters:
# 1. keep atom's POSIX shm alive across SSH logouts (logind RemoveIPC killed the runs)
# 2. NO swap: a memory-exhausted ReID must die fast (OOM -> abort -> checkpoint resume),
#    not thrash for ten hours as a zombie (observed with the 24G swapfile)
# 3. refresh the runner + resume library from this script (single source of truth)
# 4. seed ckpts/.attempts from half.log for runners that predate the attempts log
# 5. clear a failed run's RUN_DONE so reboot recovery can resume; never touch a passed run
loginctl enable-linger atom
sed -i 's/^#\?RemoveIPC=.*/RemoveIPC=no/' /etc/systemd/logind.conf
systemctl restart systemd-logind
swapoff -a 2>/dev/null; rm -f /swapfile; sed -i '/swapfile/d' /etc/fstab
mount -o remount,size=48G /dev/shm
cat > /home/atom/run_half_v2.sh.new <<'__RUNHALF_V2__'
#!/bin/bash
# One full-half GSR run, driven by instance metadata (gsr-match / gsr-half). v2: auto-resumes
# the pipeline from a per-module checkpoint if it crashes (damage control for the pytorch
# shared-memory abort that has been killing ~60% of long runs tens of hours in), instead of
# losing the whole run. Everything else -- staging, fp16 gate, scoring, guest attributes,
# self-poweroff, results layout -- is identical to run_half.sh (v1).
set -uo pipefail
export PATH="/home/atom/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME=/home/atom
export TRACKLAB_CHECKPOINT_DIR=/home/atom/ckpts
mkdir -p "$TRACKLAB_CHECKPOINT_DIR"

source /home/atom/resume_lib.sh

exec >> /home/atom/half.log 2>&1
echo "==== run_half_v2.sh boot $(date -u +%FT%TZ) ===="

# --- reboot guard: never silently restart a half-finished job ---
if [ -f /home/atom/RUN_STARTED ]; then
  if [ -f /home/atom/RUN_DONE ]; then
    echo "already done; ensuring poweroff path"
  else
    echo "REBOOT DETECTED mid-run: will resume from the newest checkpoint if the staged data survived"
    REBOOTED=1
  fi
else
  touch /home/atom/RUN_STARTED
fi
REBOOTED=${REBOOTED:-0}

MATCH=$(curl -s -H "$MH" "$MD/instance/attributes/gsr-match")
HALF=$(curl -s -H "$MH" "$MD/instance/attributes/gsr-half")
FP16=$(curl -s -H "$MH" "$MD/instance/attributes/gsr-fp16")
EXTRA_ARR=()
[ "$FP16" = "true" ] && EXTRA_ARR+=("+modules.pose_detector.service_cfg.fp16=true")
SEQ="CLPD-${MATCH}-${HALF}"
OFFSET=$(awk -F, -v m="$MATCH" -v h="$HALF" '$1==m && $2==h {print $3}' /home/atom/halfdata/frame_offsets_all_halves.csv)
if [ -z "$MATCH" ] || [ -z "$HALF" ] || [ -z "$OFFSET" ]; then
  echo "BAD CONFIG match=$MATCH half=$HALF offset=$OFFSET"; ga status "FAIL_CONFIG"; exit 1
fi
echo "seq=$SEQ offset=$OFFSET fp16=$FP16"
ga seq "$SEQ"; ga offset "$OFFSET"

# --- heartbeat every 5 min: phase + last log line + disk free ---
( while true; do
    last=$(tail -c 2000 /home/atom/pipeline.log 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-120)
    free=$(df -BG --output=avail / | tail -1 | tr -d ' G')
    ga hb "$(date -u +%FT%TZ) diskfreeG=$free ${last}"
    sleep 300
  done ) &
HB_PID=$!

finish() {  # finish <status>
  ga status "$1"
  touch /home/atom/RUN_DONE
  echo "FINAL STATUS: $1"
  # give the local watcher 2h to pull results, then power off regardless
  for i in $(seq 1 240); do
    [ -f /home/atom/RESULTS_PULLED ] && break
    sleep 30
  done
  kill $HB_PID 2>/dev/null
  sync
  sudo poweroff
}

# --- 0. reboot recovery: skip payload wait and staging when both survived ---
if [ "$REBOOTED" = 1 ]; then
  if [ -d /home/atom/SoccerNetGS/test/$SEQ/img1 ] && [ -n "$(ls -t /home/atom/ckpts/*.pklz 2>/dev/null | head -1)" ]; then
    echo "reboot recovery: staged frames and a checkpoint are present; skipping payload wait and staging"
    export RESUME_ON_START=1
  else
    echo "reboot recovery impossible (no staged frames or no checkpoint)"
    ga status "FAIL_REBOOT_MIDRUN"
    exit 1
  fi
fi

if [ "$REBOOTED" != 1 ]; then
# --- 1. wait for payload (video + GT pushed from the lab machine) ---
ga status "WAIT_PAYLOAD"
VID=/home/atom/halfdata/interim/$MATCH/${MATCH}_panorama_${HALF}_half.mp4
GT=/home/atom/halfdata/production/gsr/$MATCH/${MATCH}_${HALF}.json
for i in $(seq 1 720); do  # up to 6h
  [ -f /home/atom/halfdata/PAYLOAD_READY ] && [ -f "$VID" ] && [ -f "$GT" ] && break
  sleep 30
done
if [ ! -f /home/atom/halfdata/PAYLOAD_READY ]; then finish "FAIL_NO_PAYLOAD"; fi

# --- 2. stage with the measured offset ---
ga status "STAGING"
python3 /home/atom/stage_soccernetgs.py \
  --data /home/atom/halfdata --out-root /home/atom/SoccerNetGS \
  --match "$MATCH" --half "$HALF" --split test --frame-offset "$OFFSET" \
  > /home/atom/staging.log 2>&1
if [ $? -ne 0 ]; then tail -5 /home/atom/staging.log; finish "FAIL_STAGING"; fi
tail -3 /home/atom/staging.log
rm -f "$VID"   # ~3-7 GB back
fi  # end non-reboot branch (payload wait + staging)
NFRAMES=$(ls /home/atom/SoccerNetGS/test/$SEQ/img1 | wc -l)
ga nframes "$NFRAMES"

# --- 3. run the pipeline, auto-resuming from checkpoints on a crash ---
ga status "RUNNING"
KP=/home/atom/halfdata/keypoints
START=$(date +%s)
run_pipeline_with_resume \
  "full-$MATCH-$HALF" "$SEQ" /home/atom/SoccerNetGS \
  "$KP/${MATCH}_keypoints.json" "$KP/${MATCH}_calibrated_keypoints.json" \
  "${EXTRA_ARR[@]}"
RC=$?
WALL=$(( $(date +%s) - START ))
echo "PIPELINE_DONE rc=$RC WALL_SECONDS=$WALL resumes=$RESUME_COUNT"
ga wall_seconds "$WALL"; ga resumes "$RESUME_COUNT"

if [ -z "${PRED_JSON:-}" ]; then
  finish "FAIL_NO_PRED"
fi
PRED="$PRED_JSON"

# --- 4. score ---
ga status "SCORING"
.venv/bin/python /home/atom/smoke/score_one.py \
  --pred "$PRED" \
  --gt /home/atom/SoccerNetGS/test/$SEQ/Labels-GameState.json \
  > /home/atom/score.log 2>&1
SRC=$?
cat /home/atom/score.log
GSHOTA=$(grep -A1 'attributes ON' /home/atom/score.log | grep -oE 'GS-HOTA [0-9.]+' | awk '{print $2}')
ga gshota "${GSHOTA:-none}"

# --- 5. collect results in one place for the watcher ---
R=/home/atom/results/$SEQ
mkdir -p "$R"
cp "$PRED" "$R/pred.json"
cp /home/atom/score.log /home/atom/staging.log "$R/" 2>/dev/null
cp /home/atom/half.log "$R/half.log" 2>/dev/null
RUNDIR=$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$PRED")")")")")
cp -r "$RUNDIR/configs" "$R/configs" 2>/dev/null
cp "$RUNDIR"/eval/results/tracklab/person_summary.txt "$R/" 2>/dev/null
{ echo "seq=$SEQ"; echo "offset=$OFFSET"; echo "nframes=$NFRAMES";
  echo "pipeline_exit=$RC"; echo "wall_seconds=$WALL"; echo "score_exit=$SRC";
  echo "gshota=${GSHOTA:-none}"; echo "resumes=$RESUME_COUNT"; } > "$R/summary.txt"

if [ $SRC -ne 0 ] || [ -z "$GSHOTA" ]; then finish "FAIL_SCORING"; fi
finish "DONE_PASS_$GSHOTA"
__RUNHALF_V2__
cat > /home/atom/resume_lib.sh.new <<'__RESUME_LIB__'
#!/bin/bash
# Shared auto-resume core for run_half_v2.sh (production) and run_smoke_v2.sh (validation).
# Both source this file so the smoke drill exercises the exact same retry/checkpoint/resume
# code path that the production runner uses -- not a reimplementation of it.
#
# Requires: TRACKLAB_CHECKPOINT_DIR exported by the caller before running the pipeline.

# The published pipeline order (sn_gamestate/configs/soccertrack.yaml `pipeline:`), needed to
# turn a checkpoint's module index back into "everything still left to run".
FULL_PIPELINE=(bbox_detector pose_detector reid track jersey_number_det jersey_number_rec
               calibration gta main_subject_filter gk_role_assignment consistency
               tracklet_agg team team_side)

MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
KEEP_CKPTS=${KEEP_CKPTS:-2}

MD="http://metadata.google.internal/computeMetadata/v1"
MH="Metadata-Flavor: Google"
ga() { curl -s -X PUT --data "$2" -H "$MH" "$MD/instance/guest-attributes/gsr/$1" >/dev/null 2>&1 || true; }

# newest_checkpoint -> path or empty
#
# NOTE: checkpoints are named <video_id>-<i>-<model_name>.pklz, but video_id here is
# TrackerState's internal per-run video index (e.g. "921"), not the CLPD-<match>-<half>
# sequence name -- and model_name is the module class's own .name (e.g. "RFDETR"), not the
# pipeline config key ("bbox_detector"). Neither is predictable from the sequence name, so
# we don't try to filter by it: each run instance/attempt processes exactly one video into
# an otherwise-empty checkpoint dir, so "newest file in the dir" is unambiguous.
newest_checkpoint() {
  ls -t "$TRACKLAB_CHECKPOINT_DIR"/*.pklz 2>/dev/null | head -1
}

# tail_from_checkpoint <ckpt_path> -> space-joined module names still left to run, or empty
# Checkpoint files are named <video_id>-<i:02d>-<module_name>.pklz. video_id may itself contain
# hyphens (e.g. CLPD-128057-1st-30s), so parse the trailing two '-'-delimited fields from the
# right rather than splitting from the left.
#
# IMPORTANT: the index i is assigned by `enumerate(model_names)` in offline.py against
# whatever pipeline list THAT attempt was launched with -- it is NOT always an index into the
# full 14-module order. A first attempt (the full pipeline) numbers 0..13 against
# FULL_PIPELINE, but a resumed attempt is launched with pipeline=[tail], so ITS checkpoints
# number 0..len(tail)-1 against that shorter list. Indexing a resumed attempt's checkpoint
# into FULL_PIPELINE would compute the wrong (too-long) tail and rerun already-done modules.
# So this indexes into CURRENT_PIPELINE, which run_pipeline_with_resume keeps pointed at
# whatever module list is actually running right now.
# pipeline_for_checkpoint <ckpt_path> -> the module list the attempt that WROTE this file was
# launched with. Every attempt appends "<epoch> <modules...>" to $TRACKLAB_CHECKPOINT_DIR/.attempts
# right before launching python; the checkpoint's mtime selects the latest attempt started
# before it. This is what makes the index unambiguous even when the most recent attempt died
# before writing any checkpoint of its own (then the newest file belongs to an OLDER attempt,
# and indexing it into CURRENT_PIPELINE would skip modules -- observed 2026-09-10 as a cascade
# of KeyErrors). Falls back to the full pipeline (attempt-1 semantics) when no record exists.
pipeline_for_checkpoint() {
  local m best="" t rest
  m=$(stat -c %Y "$1")
  if [ -s "$TRACKLAB_CHECKPOINT_DIR/.attempts" ]; then
    while read -r t rest; do
      [ -n "$t" ] && [ "$t" -le "$m" ] && best="$rest"
    done < "$TRACKLAB_CHECKPOINT_DIR/.attempts"
  fi
  [ -z "$best" ] && best="${FULL_PIPELINE[*]}"
  echo "$best"
}

tail_from_checkpoint() {
  local base idx n j tail=() plist
  base=$(basename "$1" .pklz)
  idx=$(echo "$base" | rev | cut -d- -f2 | rev)
  idx=$((10#$idx))   # force base-10 (a leading zero would otherwise read as octal)
  read -r -a plist <<< "$(pipeline_for_checkpoint "$1")"
  n=${#plist[@]}
  for ((j = idx + 1; j < n; j++)); do tail+=("${plist[$j]}"); done
  echo "${tail[@]}"
}

# prune_checkpoints -- keep only the newest KEEP_CKPTS checkpoints in the dir (one video's
# worth at a time, per run instance -- see newest_checkpoint). Called between attempts
# (cheap, and checkpoints from a finished/abandoned attempt are the ones worth reclaiming);
# also run in the background every 5 min during a live attempt so a single long run can't
# pile up all 14 modules' worth (~3 GB each) at once.
prune_checkpoints() {
  ls -t "$TRACKLAB_CHECKPOINT_DIR"/*.pklz 2>/dev/null | tail -n +$((KEEP_CKPTS + 1)) | xargs -r rm -f
}

start_ckpt_pruner() {
  ( while true; do sleep 300; prune_checkpoints; done ) &
  CKPT_PRUNER_PID=$!
}

stop_ckpt_pruner() {
  [ -n "${CKPT_PRUNER_PID:-}" ] && kill "$CKPT_PRUNER_PID" 2>/dev/null
}

# run_pipeline_with_resume <experiment_name> <seq> <dataset_path> <kp_json> <kp_calib_json> [extra overrides...]
#
# Runs sn_gamestate.main; on a nonzero exit with no predictions written, finds the newest
# checkpoint for this video, computes the remaining module tail, and relaunches with
# state.load_file=<ckpt> pipeline=[tail] plus every original override. Up to MAX_ATTEMPTS
# total attempts (attempt 1 = full pipeline, the rest = resumes). On success sets PRED_JSON
# and RESUME_COUNT and returns 0; otherwise returns 1.
run_pipeline_with_resume() {
  local EXP="$1" SEQ="$2" DPATH="$3" KP="$4" KPC="$5"
  shift 5
  local EXTRA=("$@")
  local LOAD_ARGS=()
  local attempt=1
  RESUME_COUNT=0
  PRED_JSON=""
  # See tail_from_checkpoint: this must always be the module list the attempt about to run
  # (or that just ran) was actually launched with, so checkpoint indices resolve correctly
  # even across a chained resume (a resumed attempt crashing again).
  CURRENT_PIPELINE=("${FULL_PIPELINE[@]}")
  # Reboot recovery (RESUME_ON_START=1, set by run_half_v2.sh when it boots into a
  # half-finished job whose staged data and checkpoints survived): make attempt 1 itself a
  # resume from the newest checkpoint. The module list that checkpoint's index refers to is
  # read back from .current_pipeline, persisted below whenever it changes, so a checkpoint
  # written by an earlier resumed attempt still resolves to the right tail.
  if [ "${RESUME_ON_START:-0}" = 1 ]; then
    CKPT=$(newest_checkpoint)
    if [ -n "$CKPT" ]; then
      TAIL=$(tail_from_checkpoint "$CKPT")
      if [ -n "$TAIL" ]; then
        PIPE_LIST="[$(echo "$TAIL" | tr ' ' ',')]"
        LOAD_ARGS=("state.load_file=$CKPT" "pipeline=$PIPE_LIST")
        CURRENT_PIPELINE=($TAIL)
        RESUME_COUNT=1
        echo "reboot recovery: resuming from $CKPT -> pipeline=$PIPE_LIST"
      else
        echo "reboot recovery: checkpoint $CKPT already covers the whole pipeline; running the full pipeline again"
      fi
    else
      echo "reboot recovery requested but no checkpoint found; running the full pipeline"
    fi
  fi
  echo "${CURRENT_PIPELINE[*]}" > "$TRACKLAB_CHECKPOINT_DIR/.current_pipeline"
  start_ckpt_pruner
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "=== pipeline attempt $attempt/$MAX_ATTEMPTS (resume_count=$RESUME_COUNT) $(date -u +%FT%TZ) ==="
    [ ${#LOAD_ARGS[@]} -gt 0 ] && echo "    resuming with: ${LOAD_ARGS[*]}"
    echo "$(date +%s) ${CURRENT_PIPELINE[*]}" >> "$TRACKLAB_CHECKPOINT_DIR/.attempts"
    cd /home/atom/soccernet/gsr || return 1
    .venv/bin/python -u -m sn_gamestate.main -cn soccertrack \
      experiment_name="$EXP" \
      "dataset.vids_dict.test=[$SEQ]" \
      dataset.nvid=-1 \
      dataset.dataset_path="$DPATH" \
      modules.calibration.distorted_keypoints_json="$KP" \
      modules.calibration.calibrated_keypoints_json="$KPC" \
      '+modules.jersey_number_det.min_roi_area=100' \
      '+modules.jersey_number_det.min_obb_aspect_ratio=0.6' \
      modules/team=detection_level "${EXTRA[@]}" "${LOAD_ARGS[@]}" \
      visualization.cfg.save_videos=False use_rich=False num_cores=${NUM_CORES:-12} \
      >> /home/atom/pipeline.log 2>&1
    RC=$?
    PRED=$(ls -t /home/atom/soccernet/gsr/outputs/"$EXP"/*/*/eval/pred/SoccerNetGS-test/tracklab/"$SEQ".json 2>/dev/null | head -1)
    if [ -n "$PRED" ]; then
      echo "attempt $attempt produced predictions: $PRED (exit=$RC)"
      PRED_JSON="$PRED"
      prune_checkpoints
      stop_ckpt_pruner
      return 0
    fi
    if [ "$RC" -eq 0 ]; then
      echo "attempt $attempt exited 0 but wrote no predictions -- not a crash, not retrying"
      prune_checkpoints
      stop_ckpt_pruner
      return 1
    fi
    echo "attempt $attempt exited $RC with no predictions -- looking for a checkpoint to resume from"
    tail -30 /home/atom/pipeline.log | tr '\r' '\n' | tail -8
    CKPT=$(newest_checkpoint)
    if [ -z "$CKPT" ]; then
      echo "no checkpoint in $TRACKLAB_CHECKPOINT_DIR -- cannot resume"
      stop_ckpt_pruner
      return 1
    fi
    TAIL=$(tail_from_checkpoint "$CKPT")
    if [ -z "$TAIL" ]; then
      echo "checkpoint $CKPT covers the whole pipeline already -- nothing left to resume"
      stop_ckpt_pruner
      return 1
    fi
    PIPE_LIST="[$(echo "$TAIL" | tr ' ' ',')]"
    LOAD_ARGS=("state.load_file=$CKPT" "pipeline=$PIPE_LIST")
    CURRENT_PIPELINE=($TAIL)   # this is what the NEXT attempt's own checkpoint indices mean
    echo "${CURRENT_PIPELINE[*]}" > "$TRACKLAB_CHECKPOINT_DIR/.current_pipeline"
    echo "will resume from $CKPT -> pipeline=$PIPE_LIST"
    RESUME_COUNT=$((RESUME_COUNT + 1))
    prune_checkpoints
    attempt=$((attempt + 1))
  done
  echo "exhausted $MAX_ATTEMPTS attempts without predictions"
  stop_ckpt_pruner
  return 1
}
__RESUME_LIB__
cat > /home/atom/seed_attempts.py <<'__SEED__'
#!/usr/bin/env python3
"""Seed ckpts/.attempts from half.log when the runner that wrote the checkpoints predates the
attempts log (old in-memory code). Each '=== pipeline attempt N/M (...) <ISO>Z ===' line opens an
attempt; the module list is the pipeline=[...] of the following 'resuming with:' line, else the
full pipeline. Writes '<epoch> <modules...>' lines. Idempotent: no-op if .attempts is non-empty."""
import re, sys, os, datetime
FULL = "bbox_detector pose_detector reid track jersey_number_det jersey_number_rec calibration gta main_subject_filter gk_role_assignment consistency tracklet_agg team team_side"
log, out = sys.argv[1], sys.argv[2]
if os.path.exists(out) and os.path.getsize(out) > 0:
    print("attempts log already present; nothing to seed"); sys.exit(0)
txt = open(log, errors="replace").read().replace("\r", "\n")
lines = txt.split("\n")
rows = []
for i, ln in enumerate(lines):
    m = re.search(r"=== pipeline attempt \d+/\d+ .*?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z ===", ln)
    if not m: continue
    epoch = int(datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp())
    mods = FULL
    for nxt in lines[i+1:i+3]:
        pm = re.search(r"pipeline=\[([^\]]*)\]", nxt)
        if pm: mods = pm.group(1).replace(",", " "); break
    rows.append(f"{epoch} {mods}")
if not rows:
    print("no attempt lines found; nothing to seed"); sys.exit(0)
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write("\n".join(rows) + "\n")
print(f"seeded {len(rows)} attempt(s) into {out}")
__SEED__
bash -n /home/atom/run_half_v2.sh.new && bash -n /home/atom/resume_lib.sh.new && \
  mv -f /home/atom/run_half_v2.sh.new /home/atom/run_half_v2.sh && mv -f /home/atom/resume_lib.sh.new /home/atom/resume_lib.sh
chown atom:atom /home/atom/run_half_v2.sh /home/atom/resume_lib.sh /home/atom/seed_attempts.py
chmod +x /home/atom/run_half_v2.sh
mkdir -p /home/atom/ckpts && chown atom:atom /home/atom/ckpts
[ -f /home/atom/half.log ] && sudo -u atom python3 /home/atom/seed_attempts.py /home/atom/half.log /home/atom/ckpts/.attempts >> /home/atom/half.log 2>&1
if [ -f /home/atom/RUN_DONE ] && ls /home/atom/results/*/summary.txt >/dev/null 2>&1; then
  echo "startup v3: run already completed with results on disk; not restarting the runner" >> /home/atom/half.log
  exit 0
fi
if [ -f /home/atom/RUN_DONE ]; then
  echo "startup v3: previous run ended in failure; clearing RUN_DONE so recovery can resume" >> /home/atom/half.log
  rm -f /home/atom/RUN_DONE /home/atom/RESULTS_PULLED
fi
sudo -u atom env NUM_CORES=8 bash -c "nohup /home/atom/run_half_v2.sh >/dev/null 2>&1 &"

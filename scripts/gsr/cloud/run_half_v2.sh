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

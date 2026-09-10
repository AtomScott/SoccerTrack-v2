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
tail_from_checkpoint() {
  local base idx n j tail=()
  base=$(basename "$1" .pklz)
  idx=$(echo "$base" | rev | cut -d- -f2 | rev)
  idx=$((10#$idx))   # force base-10 (a leading zero would otherwise read as octal)
  n=${#CURRENT_PIPELINE[@]}
  for ((j = idx + 1; j < n; j++)); do tail+=("${CURRENT_PIPELINE[$j]}"); done
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
      if [ -s "$TRACKLAB_CHECKPOINT_DIR/.current_pipeline" ]; then
        read -r -a CURRENT_PIPELINE < "$TRACKLAB_CHECKPOINT_DIR/.current_pipeline"
      fi
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
      visualization.cfg.save_videos=False use_rich=False num_cores=12 \
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

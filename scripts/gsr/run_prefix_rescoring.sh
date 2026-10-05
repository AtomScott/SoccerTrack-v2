#!/bin/bash
# Stage labels-only GT for every cloud half with the measured offset, then rescore
# every whole-half prediction file on nested prefixes (score_prefixes.py).
set -u
REPO=/home/atom/SoccerTrack-v2
PY=/home/atom/soccernet/gsr/.venv/bin/python
OUTGT=/mnt/storage/SoccerTrack-v2/SoccerNetGS-labels
OUTJ=$REPO/results/gsr/prefix_rescoring
MOUNT=/mnt/storage/SoccerTrack-v2/gsr-cloud-results
CSV=$REPO/results/gsr/frame_offsets_all_halves.csv
mkdir -p "$OUTGT" "$OUTJ"
for root in /mnt/storage/SoccerTrack-v2/data /data/share/SoccerTrack-v2/data; do
  if [ -f "$root/production/gsr/117093/117093_1st.json" ] && [ -f "$root/interim/117093/117093_panorama_1st_half.mp4" ]; then DATA=$root; break; fi
done
echo "DATA=${DATA:-NONE} $(date -u +%FT%TZ)"
[ -z "${DATA:-}" ] && { echo "no data root with GT + video"; exit 1; }

gt_for() { # match half -> path of Labels-GameState.json (existing staged or labels-only)
  local m=$1 h=$2 seq="CLPD-$1-$2"
  case "$seq" in
    CLPD-117092-1st) echo /mnt/storage/SoccerTrack-v2/SoccerNetGS-117092/test/$seq/Labels-GameState.json; return;;
    CLPD-128057-1st|CLPD-128057-2nd) echo /mnt/storage/SoccerTrack-v2/SoccerNetGS-128057/test/$seq/Labels-GameState.json; return;;
  esac
  echo "$OUTGT/test/$seq/Labels-GameState.json"
}
pred_for() {
  local seq="CLPD-$1-$2"
  case "$seq" in
    CLPD-117092-1st) echo /mnt/storage/SoccerTrack-v2/gsr-outputs/full-117092-1st-local-a3/eval/pred/SoccerNetGS-test/tracklab/$seq.json;;
    CLPD-128057-1st) echo /mnt/storage/SoccerTrack-v2/gsr_runs/half-stageB/eval/pred/SoccerNetGS-test/tracklab/$seq.json;;
    CLPD-128057-2nd) echo /mnt/storage/SoccerTrack-v2/gsr_runs/half2-stageB-r2/eval/pred/SoccerNetGS-test/tracklab/$seq.json;;
    *) for f in gsr4 gsr2; do [ -f "$MOUNT/$f/$seq/pred.json" ] && { echo "$MOUNT/$f/$seq/pred.json"; return; }; done; echo MISSING;;
  esac
}

# 1. labels-only staging for the cloud halves
for m in 117092 117093 118575 118576 118577 118578 128057 128058 132831 132877; do
  for h in 1st 2nd; do
    seq="CLPD-$m-$h"; gt=$(gt_for $m $h)
    [ -f "$gt" ] && { echo "GT ok: $seq"; continue; }
    off=$(awk -F, -v m="$m" -v h="$h" '$1==m && $2==h {print $3}' "$CSV")
    echo "staging labels $seq offset=$off $(date -u +%T)"
    python3 $REPO/scripts/gsr/stage_soccernetgs.py --data "$DATA" --out-root "$OUTGT" --match $m --half $h --split test --frame-offset "$off" --skip-frames > "$OUTGT/stage_$seq.log" 2>&1 || { echo "STAGE FAIL $seq"; tail -3 "$OUTGT/stage_$seq.log"; }
  done
done

# 2. rescoring, two halves at a time
jobs_file=$(mktemp)
for m in 117092 117093 118575 118576 118577 118578 128057 128058 132831 132877; do
  for h in 1st 2nd; do
    seq="CLPD-$m-$h"; gt=$(gt_for $m $h); pred=$(pred_for $m $h)
    [ -f "$gt" ] || { echo "SKIP $seq: no GT"; continue; }
    [ -f "$pred" ] || { echo "SKIP $seq: no pred ($pred)"; continue; }
    [ -f "$OUTJ/$seq.json" ] && { echo "done already: $seq"; continue; }
    echo "$PY $REPO/scripts/gsr/score_prefixes.py --pred $pred --gt $gt --seq $seq --out $OUTJ/$seq.json > $OUTJ/$seq.log 2>&1; echo FINISHED $seq \$(date -u +%T)" >> "$jobs_file"
  done
done
echo "$(wc -l < "$jobs_file") scoring jobs $(date -u +%FT%TZ)"
xargs -P 2 -I CMD bash -c CMD < "$jobs_file"
echo "PREFIX_RESCORING_DONE $(date -u +%FT%TZ)"

#!/bin/bash
# Watcher for the v2 auto-resume rerun fleet (gsr4-*): 10 halves, all in Tokyo
# (asia-northeast1-a, 16 free L4 slots). Same contract as watcher.sh/watcher2.sh;
# results under gsr4/ subdir, table in STATUS3.txt. Adds the resumes= count to the
# status line since that's the whole point of this fleet.
set -u
export PATH=$PATH:$HOME/google-cloud-sdk/bin
OUT=/mnt/storage/SoccerTrack-v2/gsr-cloud-results/gsr4
mkdir -p "$OUT"
Z="asia-northeast1-a"
HALVES="117092-1st 117092-2nd 117093-1st 117093-2nd 118576-1st 118576-2nd 118577-1st 118577-2nd 118578-2nd 128058-2nd 132877-1st 132831-1st"

ga() { gcloud compute instances get-guest-attributes "$1" --zone="$Z" --query-path="gsr/$2" --format="value(value)" 2>/dev/null | head -1; }

while true; do
  TS=$(date +"%F %T JST")
  TMP=$(mktemp)
  echo "GSR v2 (auto-resume) rerun fleet status  (updated $TS)  [12 halves, Tokyo]" > "$TMP"
  printf "%-16s %-24s %-10s %-8s %s\n" HALF STATUS GS-HOTA RESUMES HEARTBEAT >> "$TMP"
  alldone=1
  for HH in $HALVES; do
    I="gsr4-$HH"
    MARK="$OUT/.handled-$I"
    if [ -f "$MARK" ]; then
      printf "%-16s %-24s %-10s %-8s %s\n" "$HH" "$(cat $MARK)" "$(cat $OUT/CLPD-${HH}/gshota 2>/dev/null)" "$(cat $OUT/CLPD-${HH}/resumes 2>/dev/null)" "results pulled" >> "$TMP"
      continue
    fi
    alldone=0
    ST=$(ga "$I" status); HB=$(ga "$I" hb); GS=$(ga "$I" gshota); RS=$(ga "$I" resumes)
    STATE=$(gcloud compute instances describe "$I" --zone="$Z" --format='value(status)' 2>/dev/null)
    printf "%-16s %-24s %-10s %-8s %s\n" "$HH" "${ST:-<none>}/$STATE" "${GS:-}" "${RS:-}" "${HB:-}" >> "$TMP"
    case "$ST" in
      DONE_*|FAIL_*)
        SEQ="CLPD-$HH"
        mkdir -p "$OUT/$SEQ"
        timeout 900 gcloud compute scp --recurse --zone="$Z" "$I:/home/atom/results/$SEQ/*" "$OUT/$SEQ/" 2>/dev/null
        timeout 300 gcloud compute scp --zone="$Z" "$I:/home/atom/half.log" "$OUT/$SEQ/half.log" 2>/dev/null
        if [ -s "$OUT/$SEQ/summary.txt" ] || [ "$STATE" != "RUNNING" ]; then
          echo "${GS:-}" > "$OUT/$SEQ/gshota"
          echo "${RS:-}" > "$OUT/$SEQ/resumes"
          timeout 120 gcloud compute ssh "$I" --zone="$Z" --command='touch /home/atom/RESULTS_PULLED' 2>/dev/null
          sleep 5
          gcloud compute instances stop "$I" --zone="$Z" >/dev/null 2>&1
          echo "$ST" > "$MARK"
          echo "[$TS] $I -> $ST, resumes=${RS:-0}, results pulled, instance stopped" >> "$OUT/watcher.log"
        fi
        ;;
    esac
  done
  mv "$TMP" "$OUT/../STATUS3.txt"
  [ "$alldone" = 1 ] && { echo "[$TS] fleet-4 all handled; watcher3 exiting" >> "$OUT/watcher.log"; break; }
  sleep 300
done

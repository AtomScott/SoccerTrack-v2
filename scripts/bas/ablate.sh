#!/usr/bin/env bash
# Feature-group ablation for the trajectory spotter, on validation only.
#
# The point of Experiment B is not the headline number, it is WHICH PART of the game state
# carries a ball event. The four groups are:
#
#   global      where the 22 players are as a whole -- centroid, dispersion, convex hull
#   contest     the ball proxy: a soft minimum over opposing pairs, its motion, and how
#               many players are converging on it
#   team        each side's shape, speed and defensive/attacking line
#   occupancy   a coarse 6x3 soft player-density grid per team
#
# Groups are ZEROED, not removed, so every run has the same input width and the same
# parameter count. A group that turns out not to matter cannot be confused with a
# smaller model.
#
# Every number printed here is from the validation matches (117093, 132877). The test
# split is scored once, at the end, with the configuration already fixed.
#
# USAGE
#   HIDDEN=128 DROPOUT=0.2 bash scripts/bas/ablate.sh
set -euo pipefail

PY=${PY:-/home/atom/SoccerTrack-v2/.venv/bin/python}
ROOT=${ROOT:-outputs/ablate}
EPOCHS=${EPOCHS:-24}
HIDDEN=${HIDDEN:-128}
DROPOUT=${DROPOUT:-0.2}
LR=${LR:-3e-3}
mkdir -p "$ROOT"

run () {
  local name=$1; shift
  local dir="$ROOT/$name"
  mkdir -p "$dir"
  echo "=== $name : $* ==="
  $PY scripts/bas/train_trajectory.py --epochs "$EPOCHS" --steps 200 \
      --hidden "$HIDDEN" --dropout "$DROPOUT" --lr "$LR" --out "$dir" "$@" \
      > "$dir/log.txt" 2>&1 || { echo "  FAILED, see $dir/log.txt"; return 0; }
  tr '\r' '\n' < "$dir/log.txt" | grep -a 'restored epoch' | sed 's/^/  /'
}

run all
run contest_only    --keep-groups contest
run occupancy_only  --keep-groups occupancy
run team_only       --keep-groups team
run no_contest      --keep-groups global team occupancy

echo
echo "============== FEATURE ABLATION (validation only) =============="
for d in "$ROOT"/*/; do
  n=$(basename "$d")
  v=$(tr '\r' '\n' < "$d/log.txt" 2>/dev/null | grep -a 'restored epoch' \
      | sed 's/.*val mAP@1s //;s/)//' || true)
  printf '  %-16s val mAP@1s %s\n' "$n" "${v:-FAILED}"
done

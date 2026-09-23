#!/usr/bin/env bash
# Validation-only hyperparameter sweep for the trajectory spotter.
#
# WHY THIS IS A SEPARATE SCRIPT
#   The test split (128057, 132831) must be scored exactly once, at the end. Choosing a
#   configuration by looking at test numbers and then reporting them is the commonest way a
#   benchmark result becomes meaningless. Every number this script prints comes from the
#   validation matches (117093, 132877); the test predictions each run writes are left
#   unscored until a configuration has been fixed.
#
#   The first run overfits by epoch 8 even with reflection augmentation -- validation mAP@1s
#   peaks at 0.359 and then declines -- so the sweep is over capacity and regularisation
#   rather than over training length.
#
# USAGE
#   bash scripts/bas/sweep.sh                 # ~20 min on CPU
#   grep -H 'restored epoch' outputs/sweep/*/log.txt
set -euo pipefail

PY=${PY:-/home/atom/SoccerTrack-v2/.venv/bin/python}
ROOT=${ROOT:-outputs/sweep}
EPOCHS=${EPOCHS:-24}
mkdir -p "$ROOT"

run () {
  local name=$1; shift
  local dir="$ROOT/$name"
  mkdir -p "$dir"
  echo "=== $name : $* ==="
  $PY scripts/bas/train_trajectory.py --epochs "$EPOCHS" --steps 200 \
      --out "$dir" "$@" > "$dir/log.txt" 2>&1 || { echo "  FAILED, see $dir/log.txt"; return 0; }
  grep -a 'restored epoch' "$dir/log.txt" | sed 's/^/  /'
  grep -a 'chosen on validation' "$dir/log.txt" | sed 's/^/  /'
}

run h128_d02   --hidden 128 --dropout 0.2
run h128_d04   --hidden 128 --dropout 0.4
run h64_d02    --hidden 64  --dropout 0.2
run h64_d04    --hidden 64  --dropout 0.4
run h96_d03    --hidden 96  --dropout 0.3
run h64_d04_lo --hidden 64  --dropout 0.4 --lr 1e-3
# Ablation, not a candidate: shows what the reflection augmentation is worth.
run h64_d04_noaug --hidden 64 --dropout 0.4 --no-augment

echo
echo "===================== SUMMARY (validation only) ====================="
for d in "$ROOT"/*/; do
  n=$(basename "$d")
  v=$(grep -a 'restored epoch' "$d/log.txt" 2>/dev/null | sed 's/.*val mAP@1s //;s/)//' || true)
  e=$(grep -a 'restored epoch' "$d/log.txt" 2>/dev/null | sed 's/.*restored epoch \([0-9]*\).*/\1/' || true)
  c=$(grep -a 'chosen on validation' "$d/log.txt" 2>/dev/null | sed 's/.*floor=//;s/ (mAP.*//' || true)
  printf '  %-18s val mAP@1s %-8s at epoch %-4s decode %s\n' "$n" "${v:-FAILED}" "${e:--}" "${c:--}"
done

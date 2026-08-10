#!/usr/bin/env bash
# Download SoccerTrack v2 from Hugging Face into a local directory.
#
# Usage:
#   scripts/download.sh [--dest DIR] [--revision REV] [--match ID ...] [--include PATTERN ...]
#
#   --dest DIR       Target directory (default: ./data)
#   --revision REV   Pin to a specific dataset revision / tag / commit (default: main)
#   --match ID       Repeatable. Restrict download to one or more match IDs
#                    (e.g. --match 117092 --match 117093). Selects gsr/<id>, bas/<id>,
#                    mot/<id>, raw/<id>, videos/<id>*. Mutually exclusive with --include.
#   --include PAT    Repeatable raw include pattern forwarded to huggingface-cli.
#
# Requirements:
#   - The Hugging Face CLI (`hf` in newer versions, `huggingface-cli` in older).
#     Install with: pip install -U huggingface_hub
#   - For gated / private access: `hf auth login` (or HF_TOKEN env var).
set -euo pipefail

REPO="atomscott/soccertrack-v2"
DEST="./data"
REVISION="main"
MATCHES=()
INCLUDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)      DEST="$2"; shift 2 ;;
    --revision)  REVISION="$2"; shift 2 ;;
    --match)     MATCHES+=("$2"); shift 2 ;;
    --include)   INCLUDES+=("$2"); shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#MATCHES[@]} -gt 0 && ${#INCLUDES[@]} -gt 0 ]]; then
  echo "--match and --include are mutually exclusive." >&2
  exit 2
fi

if command -v hf >/dev/null 2>&1; then
  CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  CLI=(huggingface-cli download)
else
  echo "huggingface CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

ARGS=("${CLI[@]}" --repo-type dataset --revision "$REVISION" --local-dir "$DEST" "$REPO")

for m in "${MATCHES[@]}"; do
  ARGS+=(--include "gsr/${m}/*" --include "bas/${m}/*" --include "mot/${m}/*" \
         --include "raw/${m}/*" --include "videos/${m}*")
done
for pat in "${INCLUDES[@]}"; do
  ARGS+=(--include "$pat")
done

mkdir -p "$DEST"
echo "[download] ${ARGS[*]}"
exec "${ARGS[@]}"

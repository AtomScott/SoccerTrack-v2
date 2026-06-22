#!/usr/bin/env bash
# Download SoccerTrack v2 from Hugging Face into a local directory.
#
# Usage:
#   scripts/download.sh [--dest DIR] [--revision REV] [--match ID ...] [--include PATTERN ...]
#
#   --dest DIR       Target directory (default: ./data)
#   --revision REV   Pin to a specific dataset revision / tag / commit (default: main)
#   --match ID       Repeatable. Restrict download to one or more match IDs
#                    (e.g. --match 117099 --match 117100). Selects gsr/<id>, bas/<id>,
#                    mot/<id>, raw/<id>, videos/<id>*. Mutually exclusive with --include.
#   --include PAT    Repeatable raw include pattern forwarded to huggingface-cli.
#
# Requirements:
#   - The Hugging Face CLI (`hf` in newer versions, `huggingface-cli` in older).
#     Install with: pip install -U huggingface_hub  (it is also a declared
#     dependency of this repo, so `uv sync` / activating .venv provides it).
#
# Authentication (REQUIRED):
#   The SoccerTrack v2 dataset (atomscott/soccertrack-v2) is gated. You MUST be
#   authenticated to Hugging Face before downloading. Provide credentials in
#   EITHER way:
#     1. Export a token:   export HF_TOKEN=hf_xxx        (recommended for CI)
#     2. Interactive login: hf auth login                (stores a token on disk)
#   This script PREFLIGHTS the credential and exits with a clear error if neither
#   is present — it will not hand an unauthenticated request to the CLI.
#   Get / manage tokens at https://huggingface.co/settings/tokens and request
#   access to the dataset at https://huggingface.co/datasets/atomscott/soccertrack-v2
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
      sed -n '2,28p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#MATCHES[@]} -gt 0 && ${#INCLUDES[@]} -gt 0 ]]; then
  echo "--match and --include are mutually exclusive." >&2
  exit 2
fi

if command -v hf >/dev/null 2>&1; then
  HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN=huggingface-cli
else
  echo "huggingface CLI not found. Install with: pip install -U huggingface_hub" >&2
  echo "(Or activate this repo's venv: source .venv/bin/activate)" >&2
  exit 1
fi

# --- Authentication preflight -------------------------------------------------
# The dataset is gated: fail loudly NOW if no credential is available, rather
# than letting the CLI emit an opaque 401/403 mid-download.
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[download] using HF_TOKEN from environment."
else
  # No env token — check for a stored login (`hf auth login` / `huggingface-cli login`).
  if "$HF_BIN" auth whoami >/dev/null 2>&1; then
    echo "[download] using stored Hugging Face login ($("$HF_BIN" auth whoami 2>/dev/null | head -n1))."
  elif "$HF_BIN" whoami >/dev/null 2>&1; then
    # Older CLIs expose `whoami` at the top level instead of under `auth`.
    echo "[download] using stored Hugging Face login ($("$HF_BIN" whoami 2>/dev/null | head -n1))."
  else
    echo "ERROR: not authenticated to Hugging Face." >&2
    echo "  atomscott/soccertrack-v2 is gated. Authenticate with EITHER:" >&2
    echo "    export HF_TOKEN=hf_xxx        # token from https://huggingface.co/settings/tokens" >&2
    echo "    ${HF_BIN} auth login          # interactive login (stores a token)" >&2
    echo "  and request dataset access at" >&2
    echo "    https://huggingface.co/datasets/atomscott/soccertrack-v2" >&2
    exit 1
  fi
fi

ARGS=("$HF_BIN" download --repo-type dataset --revision "$REVISION" --local-dir "$DEST" "$REPO")

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

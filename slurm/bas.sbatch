#!/bin/bash
# Slurm batch script: BAS baseline (train -> eval) for SoccerTrack v2.
#
# Submit from the repo root so that `python -m baselines.bas.*` resolves
# (the module path requires the repo root to be on sys.path, which `python -m`
# guarantees only when the current working directory is the repo root):
#
#     cd /home/atom/soccertrack-v2 && sbatch slurm/bas.sbatch
#
# This script `cd`s into $REPO below, so it is safe to submit from anywhere; the
# note above is for running the `python -m ...` lines by hand.
#
# The logs/ directory must exist at submit time — Slurm does NOT create the
# --output/--error directory itself and the job will fail to launch if it is
# missing. It is kept in version control via logs/.gitkeep; if you relocate the
# repo, run `mkdir -p logs` once before submitting.

#SBATCH --job-name=stv2-bas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home/atom/soccertrack-v2/logs/%x-%j.out
#SBATCH --error=/home/atom/soccertrack-v2/logs/%x-%j.err

# --- Node selection -----------------------------------------------------------
# The `gpu` partition has three nodes: ctrl, gpu01, gpu02 (each advertises
# gpu:1). As of this writing the dedicated workers gpu01 and gpu02 are DOWN
# ("Not responding"), and ctrl (the head node) is the only idle GPU node.
# We deliberately do NOT pin or hard-exclude any node: with the default
# scheduler the job will land on whatever GPU node is up — which today means
# ctrl. Uncomment ONE of the lines below to override once the workers recover:
#
#   # Prefer the dedicated workers, avoid running on the controller:
#   #SBATCH --exclude=ctrl
#
#   # Or pin to specific node(s):
#   #SBATCH --nodelist=gpu01
#
# (Leaving both commented lets Slurm place the job on any available gpu node,
#  including ctrl, so the job is not blocked while gpu01/gpu02 are down.)

set -euo pipefail

REPO=/home/atom/soccertrack-v2
cd "$REPO"

# --- Environment --------------------------------------------------------------
source "$REPO/.venv/bin/activate"

# --- Preflight probe ----------------------------------------------------------
echo "===== preflight ($(date -Is)) ====="
echo "host:        $(hostname)"
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID:-<none>}"
echo "SLURM_NODELIST:      ${SLURM_NODELIST:-<none>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "python:      $(command -v python)"
nvidia-smi || { echo "ERROR: nvidia-smi failed — no usable GPU on this node." >&2; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: torch.cuda.is_available() is False — aborting before training." >&2; exit 1; }
echo "===== preflight ok ====="

# --- Train --------------------------------------------------------------------
echo "===== train ($(date -Is)) ====="
python -m baselines.bas.train --config baselines/bas/config.yaml

# --- Eval ---------------------------------------------------------------------
echo "===== eval ($(date -Is)) ====="
python -m baselines.bas.eval --config baselines/bas/config.yaml

echo "===== done ($(date -Is)) ====="

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
PY="/home/introai11/.conda/envs/openpi_env/bin/python"

NUM_ACTIONS=$((1 << 20))
ACTION_DIM=32
CODE_BITS=24
TRAIN_SAMPLES=65536
EVAL_SAMPLES=131072
SEED=7
OUTPUT_DIR="${PROJECT_ROOT}/experiments/results/orthogonality_correlation_sweep"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-actions) NUM_ACTIONS="$2"; shift 2 ;;
    --action-dim) ACTION_DIM="$2"; shift 2 ;;
    --code-bits) CODE_BITS="$2"; shift 2 ;;
    --train-samples) TRAIN_SAMPLES="$2"; shift 2 ;;
    --eval-samples) EVAL_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/experiments:${PYTHONPATH:-}"

"${PY}" "${PROJECT_ROOT}/experiments/orthogonality_correlation_sweep.py" \
  --num-actions "${NUM_ACTIONS}" \
  --action-dim "${ACTION_DIM}" \
  --code-bits "${CODE_BITS}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}"

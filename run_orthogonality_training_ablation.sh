#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
PY="/home/introai11/.conda/envs/openpi_env/bin/python"

DEVICE="cpu"
OUTPUT_DIR="${PROJECT_ROOT}/experiments/results/orthogonality_training_ablation"
STEPS=500
TRAIN_SAMPLES=6000
VAL_SAMPLES=6000
BATCH_SIZE=512
INPUT_DIM=32
NUM_ACTIONS=4096
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --train-samples) TRAIN_SAMPLES="$2"; shift 2 ;;
    --val-samples) VAL_SAMPLES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --input-dim) INPUT_DIM="$2"; shift 2 ;;
    --num-actions) NUM_ACTIONS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PY}" "${PROJECT_ROOT}/experiments/train_orthogonality_ablation.py" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --input-dim "${INPUT_DIM}" \
  --num-actions "${NUM_ACTIONS}" \
  --seed "${SEED}"

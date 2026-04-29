#!/usr/bin/env bash
# =============================================================================
# run_finetune_aloha.sh
# Wrapper for Octo ALOHA finetune using the local multi-GPU script.
# Run inside the Singularity (vla) container.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
PY="/home/introai11/.conda/envs/octo_jax39/bin/python"
ENV_PREFIX="/home/introai11/.conda/envs/octo_jax39"
OCTO_DIR="${PROJECT_ROOT}/third_party/octo"

PRETRAINED_PATH="hf://rail-berkeley/octo-small-1.5"
DATA_DIR="${PROJECT_ROOT}/data/aloha_sim"
SAVE_DIR="${PROJECT_ROOT}/checkpoints/octo_aloha"
BATCH_SIZE=128
STEPS=5000
SAVE_INTERVAL=1000
ACTION_HORIZON=50
WINDOW_SIZE=1
NUM_DEVICES=1
FREEZE_TRANSFORMER=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --pretrained_path) PRETRAINED_PATH="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --save_dir) SAVE_DIR="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --save_interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --action_horizon) ACTION_HORIZON="$2"; shift 2 ;;
    --window_size) WINDOW_SIZE="$2"; shift 2 ;;
    --num_devices) NUM_DEVICES="$2"; shift 2 ;;
    --freeze_transformer) FREEZE_TRANSFORMER=true; shift ;;
    *) echo "Unknown arg: $1"; shift ;;
  esac
done

mkdir -p "${SAVE_DIR}"

if [ -d "${DATA_DIR}/aloha_sim_cube_scripted_dataset" ]; then
  DATA_DIR="${DATA_DIR}"
elif [ -d "${DATA_DIR}/aloha_sim_dataset/aloha_sim_cube_scripted_dataset" ]; then
  DATA_DIR="${DATA_DIR}/aloha_sim_dataset"
fi

if [ ! -d "${DATA_DIR}/aloha_sim_cube_scripted_dataset" ]; then
  echo "Expected dataset at:"
  echo "  ${DATA_DIR}/aloha_sim_cube_scripted_dataset"
  echo "but it was not found."
  exit 1
fi

export PYTHONPATH="${OCTO_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PATH="${ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "${NUM_DEVICES}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=0
elif [ "${NUM_DEVICES}" = "2" ]; then
  export CUDA_VISIBLE_DEVICES=0,1
fi

echo "=== Octo ALOHA Finetune (multi-GPU wrapper) ==="
echo "  Pretrained : ${PRETRAINED_PATH}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Save dir   : ${SAVE_DIR}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Steps      : ${STEPS}"
echo "  Save every : ${SAVE_INTERVAL}"
echo "  Window     : ${WINDOW_SIZE}"
echo "  Horizon    : ${ACTION_HORIZON}"
echo "  Requested  : ${NUM_DEVICES}"
echo "  Devices    : $(${PY} -c 'import jax; print(jax.device_count())')"
echo ""

${PY} "${PROJECT_ROOT}/experiments/finetune_aloha_multigpu.py" \
  --pretrained_path="${PRETRAINED_PATH}" \
  --data_dir="${DATA_DIR}" \
  --save_dir="${SAVE_DIR}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_interval="${SAVE_INTERVAL}" \
  --window_size="${WINDOW_SIZE}" \
  --action_horizon="${ACTION_HORIZON}" \
  --num_devices="${NUM_DEVICES}" \
  $([ "${FREEZE_TRANSFORMER}" = true ] && echo "--freeze_transformer" || echo "")

echo ""
echo "Finetune complete. Checkpoints saved to: ${SAVE_DIR}"

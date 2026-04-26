#!/usr/bin/env bash
set -euo pipefail

source /tools/anaconda3/etc/profile.d/conda.sh
conda activate openpi_env

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

/home/introai11/.conda/envs/openpi_env/bin/python \
  "${PROJECT_ROOT}/experiments/pi05_verified_gvla_benchmark.py" \
  --device cuda \
  --dtype bfloat16 \
  "$@"

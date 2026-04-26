#!/usr/bin/env bash
set -euo pipefail

source /tools/anaconda3/etc/profile.d/conda.sh
conda activate vla

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

/home/introai11/.conda/envs/vla/bin/python \
  "${PROJECT_ROOT}/experiments/action_codebook_collision_test.py" \
  "$@"

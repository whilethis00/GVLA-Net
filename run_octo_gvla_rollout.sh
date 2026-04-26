#!/usr/bin/env bash
set -euo pipefail

source /tools/anaconda3/etc/profile.d/conda.sh
conda activate octo_env

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
ENV_SITE_PACKAGES="/home/introai11/.conda/envs/octo_env/lib/python3.10/site-packages"
export PYTHONPATH="${ENV_SITE_PACKAGES}:${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/octo:${PYTHONPATH:-}"

/home/introai11/.conda/envs/octo_env/bin/python \
  "${PROJECT_ROOT}/experiments/octo_gvla_rollout.py" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

source /tools/anaconda3/etc/profile.d/conda.sh
conda activate octo_jax39

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
ENV_SITE_PACKAGES="/home/introai11/.conda/envs/octo_jax39/lib/python3.9/site-packages"
export PATH="/home/introai11/.conda/envs/octo_jax39/bin:${PATH}"
export LD_LIBRARY_PATH="/home/introai11/.conda/envs/octo_jax39/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ENV_SITE_PACKAGES}:${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/octo:${PYTHONPATH:-}"

/home/introai11/.conda/envs/octo_jax39/bin/python \
  "${PROJECT_ROOT}/experiments/octo_gvla_rollout.py" \
  "$@"

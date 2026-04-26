#!/usr/bin/env bash
set -euo pipefail

source /tools/anaconda3/etc/profile.d/conda.sh
conda activate vla

python3 /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/sota_vla_integration.py \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  "$@"

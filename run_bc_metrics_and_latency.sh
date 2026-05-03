#!/usr/bin/env bash
set -euo pipefail

cd /home/introai11/.agile/users/hsjung/projects/GVLA-Net

/home/introai11/.conda/envs/vla/bin/python experiments/bc_validation_metrics.py \
  --checkpoints-root experiments/results/bc_study/checkpoints \
  --output-dir experiments/results/bc_study/validation_metrics \
  --m-values 128 256 1024 2048 \
  --device cuda

/home/introai11/.conda/envs/vla/bin/python experiments/bc_end_to_end_latency.py \
  --checkpoints-root experiments/results/bc_study/checkpoints \
  --output-dir experiments/results/bc_study/end_to_end_latency \
  --m-values 128 256 1024 2048 \
  --batch-sizes 1 256 \
  --device cuda

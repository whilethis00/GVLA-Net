#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
RUNNER="${PROJECT_ROOT}/run_octo_gvla_rollout.sh"

run() {
  echo
  echo "==> $*"
  "${RUNNER}" "$@"
}

run --rollouts 10 --seed 0 --run-name octo_seed0_r10
run --rollouts 10 --seed 1 --run-name octo_seed1_r10
run --rollouts 10 --seed 2 --run-name octo_seed2_r10
run --rollouts 10 --seed 0 --gvla-bins 4096 --run-name octo_gvla_4k_s0
run --rollouts 10 --seed 0 --gvla-bins 65536 --run-name octo_gvla_64k_s0
run --rollouts 10 --seed 0 --gvla-bins 1048576 --run-name octo_gvla_1m_s0

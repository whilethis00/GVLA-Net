#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
PY="/home/introai11/.conda/envs/dhmamba/bin/python"
SCRIPT="$ROOT/experiments/robosuite_quantization_study.py"
RESULTS="$ROOT/experiments/results"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_case() {
  local name="$1"
  shift
  local save_dir="$RESULTS/$name"

  if [ -f "$save_dir/results.json" ]; then
    echo "[skip] $name already finished"
    return
  fi

  mkdir -p "$save_dir"
  echo "[start] $name"
  "$PY" "$SCRIPT" \
    --task pick_place_can \
    --decode_mode ar_tokens \
    --n_rollouts 50 \
    --max_steps 400 \
    --device cpu \
    --skip_latency \
    --save_dir "$save_dir" \
    "$@"
  echo "[done] $name"
}

cd "$ROOT"

# Latency stress: AR accumulates 7 token decodes per action.
run_case robosuite_ar_compare_ar50_queue --ar_token_bins 256 --token_latency_ms 50 --token_error_prob 0.0
run_case robosuite_ar_compare_ar75_queue --ar_token_bins 256 --token_latency_ms 75 --token_error_prob 0.0
run_case robosuite_ar_compare_ar100_queue --ar_token_bins 256 --token_latency_ms 100 --token_error_prob 0.0

# Stronger token corruption.
run_case robosuite_ar_compare_err020_queue --ar_token_bins 256 --token_latency_ms 0 --token_error_prob 0.20
run_case robosuite_ar_compare_err030_queue --ar_token_bins 256 --token_latency_ms 0 --token_error_prob 0.30

# Lower-resolution AR to probe whether it also needs high bin counts.
run_case robosuite_ar_compare_ar128_queue --ar_token_bins 128 --token_latency_ms 0 --token_error_prob 0.0
run_case robosuite_ar_compare_ar128_lat50_queue --ar_token_bins 128 --token_latency_ms 50 --token_error_prob 0.0
run_case robosuite_ar_compare_ar96_queue --ar_token_bins 96 --token_latency_ms 0 --token_error_prob 0.0
run_case robosuite_ar_compare_ar96_lat50_queue --ar_token_bins 96 --token_latency_ms 50 --token_error_prob 0.0

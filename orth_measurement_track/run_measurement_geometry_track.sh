#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
PYTHON_BIN="${PYTHON_BIN:-/home/introai11/.conda/envs/vla/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUT_BASE="${OUT_BASE:-$ROOT/experiments/results/orth_measurement_track}"

mkdir -p "$OUT_BASE"

cd "$ROOT"

echo "[1/4] target geometry diagnostic"
"$PYTHON_BIN" experiments/bc_target_bit_transition_diagnostic.py \
  --n-bins 1024 \
  --output-dir "$OUT_BASE/target_geometry"

echo "[2/4] proxy measurement correlation sweep"
"$PYTHON_BIN" experiments/orthogonality_correlation_sweep.py \
  --output-dir "$OUT_BASE/proxy_corr_sweep" \
  --run-name "proxy_corr_sweep"

echo "[3/4] trainable orthogonality ablation"
"$PYTHON_BIN" experiments/train_orthogonality_ablation.py \
  --device "$DEVICE" \
  --output-dir "$OUT_BASE/trainable_ortho_sweep"

echo "[4/4] BC measurement geometry ablation"
"$PYTHON_BIN" orth_measurement_track/bc_measurement_geometry_ablation.py \
  --device "$DEVICE" \
  --output-dir "$OUT_BASE/bc_measurement_geometry_ablation"

cat <<EOF

Saved separate-track outputs under:
  $OUT_BASE

This suite intentionally avoids:
  - experiments/results/bc_study/
  - results/

Missing BC-specific experiment for this track:
  optional larger-seed / larger-rollout reruns for the BC ablation
EOF

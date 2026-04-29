#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/introai11/.agile/users/hsjung/projects/GVLA-Net
PYTHON=/home/introai11/.conda/envs/dhmamba/bin/python
DATA=$ROOT/data/robomimic/lift/ph/low_dim_v141.hdf5
CKPT_DIR=$ROOT/experiments/results/bc_study/checkpoints
EVAL_OUT=$ROOT/experiments/results/bc_study/eval_results_2048.json
PLOT_DIR=$ROOT/experiments/results/bc_study/figures_2048

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT"

echo "============================================"
echo " BC Gray-code 2048 study"
echo "============================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo ""
echo "[TRAIN] dense_2048"
$PYTHON experiments/bc_train.py \
  --data "$DATA" \
  --head dense \
  --n_bins 2048 \
  --epochs 200 \
  --batch_size 256 \
  --lr 3e-4 \
  --exp_name dense_2048 \
  --out_dir "$CKPT_DIR"

echo ""
echo "[TRAIN] gvla_2048"
$PYTHON experiments/bc_train.py \
  --data "$DATA" \
  --head gvla \
  --n_bins 2048 \
  --epochs 200 \
  --batch_size 256 \
  --lr 3e-4 \
  --exp_name gvla_2048 \
  --out_dir "$CKPT_DIR"

echo ""
echo "[TRAIN] gvla_gray_2048"
$PYTHON experiments/bc_train.py \
  --data "$DATA" \
  --head gvla \
  --n_bins 2048 \
  --epochs 200 \
  --batch_size 256 \
  --lr 3e-4 \
  --exp_name gvla_gray_2048 \
  --out_dir "$CKPT_DIR" \
  --gray_code

echo ""
echo "[EVAL] rollout only"
$PYTHON experiments/bc_eval.py \
  --sweep_dir "$CKPT_DIR" \
  --n_rollouts 50 \
  --max_steps 500 \
  --out "$EVAL_OUT" \
  --plot_dir "$PLOT_DIR" \
  --skip_latency

echo ""
echo "Done."
echo "Checkpoints: $CKPT_DIR"
echo "Eval JSON:   $EVAL_OUT"
echo "Figures:     $PLOT_DIR"

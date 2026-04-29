#!/usr/bin/env bash
set -eu

PYTHON=/home/introai11/.conda/envs/octo_jax39/bin/python
SCRIPT=/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py
CKPT_ROOT=/home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha

MODE=${1:-debug}

case "$MODE" in
  debug)
    MUJOCO_GL=egl $PYTHON $SCRIPT \
      --checkpoint_path $CKPT_ROOT/4999 \
      --task AlohaTransferCube-v0 \
      --rollouts 1 \
      --max_steps 400 \
      --seed 0 \
      --debug_action_steps 10 \
      --debug_save_frames
    ;;
  sweep)
    for ckpt in 999 1999 2999 3999 4999; do
      echo "===== CKPT $ckpt ====="
      MUJOCO_GL=egl $PYTHON $SCRIPT \
        --checkpoint_path $CKPT_ROOT/$ckpt \
        --task AlohaTransferCube-v0 \
        --rollouts 5 \
        --max_steps 400 \
        --seed 0
    done
    ;;
  full)
    MUJOCO_GL=egl $PYTHON $SCRIPT \
      --checkpoint_path $CKPT_ROOT/4999 \
      --task AlohaTransferCube-v0 \
      --rollouts 50 \
      --max_steps 400 \
      --seed 0
    ;;
  *)
    echo "Usage: $0 [debug|sweep|full]"
    exit 1
    ;;
esac

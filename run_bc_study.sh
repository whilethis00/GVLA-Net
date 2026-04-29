#!/usr/bin/env bash
# run_bc_study.sh
# ================
# Full BC study: train Dense & GVLA heads at M ∈ {8,16,32,64,128,256,512,1024}
# then evaluate success rate + latency and produce paper figures.
#
# Usage:
#   bash run_bc_study.sh [--fast]   # --fast uses fewer epochs for quick test

set -e

PYTHON=/home/introai11/.conda/envs/dhmamba/bin/python
DATA=data/robomimic/lift/ph/low_dim_v141.hdf5
CKPT_DIR=experiments/results/bc_study/checkpoints
EVAL_OUT=experiments/results/bc_study/eval_results.json
PLOT_DIR=experiments/results/bc_study/figures

EPOCHS=200
N_ROLLOUTS=50

if [[ "$1" == "--fast" ]]; then
    EPOCHS=50
    N_ROLLOUTS=20
    echo "[fast mode] epochs=$EPOCHS, rollouts=$N_ROLLOUTS"
fi

cd /home/introai11/.agile/users/hsjung/projects/GVLA-Net

echo "============================================"
echo " GVLA-Net BC Study: Dense vs GVLA heads"
echo "============================================"

# ── Training sweep ────────────────────────────────────────────────────────
# dense & gvla (natural binary)
for HEAD in dense gvla; do
    for M in 8 16 32 64 128 256 512 1024; do
        EXP="${HEAD}_${M}"
        echo ""
        echo "[TRAIN] head=$HEAD  M=$M  exp=$EXP"
        $PYTHON experiments/bc_train.py \
            --data "$DATA" \
            --head "$HEAD" \
            --n_bins "$M" \
            --epochs "$EPOCHS" \
            --exp_name "$EXP" \
            --out_dir "$CKPT_DIR"
    done
done

# gvla + gray code (ablation)
for M in 8 16 32 64 128 256 512 1024; do
    EXP="gvla_gray_${M}"
    echo ""
    echo "[TRAIN] head=gvla  M=$M  gray_code=True  exp=$EXP"
    $PYTHON experiments/bc_train.py \
        --data "$DATA" \
        --head gvla \
        --n_bins "$M" \
        --epochs "$EPOCHS" \
        --exp_name "$EXP" \
        --out_dir "$CKPT_DIR" \
        --gray_code
done

echo ""
echo "============================================"
echo " Evaluation: rollout + latency"
echo "============================================"

$PYTHON experiments/bc_eval.py \
    --sweep_dir "$CKPT_DIR" \
    --n_rollouts "$N_ROLLOUTS" \
    --max_steps 500 \
    --out "$EVAL_OUT" \
    --plot_dir "$PLOT_DIR"

echo ""
echo "Done. Results: $EVAL_OUT"
echo "Figures:       $PLOT_DIR"

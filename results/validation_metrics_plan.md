# Validation Metrics

## Goal

Add mechanism-level evidence that supports:

> Gray code improves the learnability of bitwise high-resolution action heads by improving code-space locality.

## Required Metrics

- action `L1`
- action `L2`
- bin error `|b_hat - b|`
- Hamming error
- adjacent-bin error rate

## Current Status

Saved validation-metric artifacts now exist.

Produced files:

- `experiments/results/bc_study/validation_metrics/validation_split_lock.json`
- `experiments/results/bc_study/validation_metrics/validation_metrics.csv`
- `experiments/results/bc_study/validation_metrics/validation_metrics.json`
- `experiments/results/bc_study/validation_metrics/validation_metrics.md`
- `experiments/results/bc_study/validation_metrics/predictions/*.csv`

Exporter:

- `experiments/bc_validation_metrics.py`

## Locked Validation Source

Use the saved BC dataset path from checkpoint configs:

- `data/robomimic/lift/ph/low_dim_v141.hdf5`

Locked split rule:

- deterministic shuffled split
- `split_seed=20260503`
- `val_fraction=0.10`
- one shared index list reused for every run

Saved split artifact:

- `experiments/results/bc_study/validation_metrics/validation_split_lock.json`
- `results/validation_split_lock.md`

## Covered Runs

The saved artifact covers:

- `Dense CE, M=128/256/1024/2048`
- `Bitwise Natural, M=128/256/1024/2048`
- `Bitwise Gray, M=128/256/1024/2048`

Saved prediction fields:

- sample id
- ground-truth continuous action
- predicted continuous action
- ground-truth bin
- predicted bin
- predicted bits if available
- target bits if available

Prediction directory:

- `experiments/results/bc_study/validation_metrics/predictions/<run_name>.csv`

## Saved Metrics

Each summary row includes:

- mean action `L1`
- mean action `L2`
- mean bin error
- mean Hamming error
- adjacent-bin error rate

Summary files:

- `experiments/results/bc_study/validation_metrics/validation_metrics.csv`
- `experiments/results/bc_study/validation_metrics/validation_metrics.json`
- `experiments/results/bc_study/validation_metrics/validation_metrics.md`

## Locked Interpretation

What the artifact shows:

- Gray beats natural binary on every tested `M` for:
  - action `L1`
  - action `L2`
  - mean bin error
  - mean Hamming error
- Example at `M=2048`:
  - Natural: `L1=0.0551`, `bin error=56.37`, `Hamming=0.2030`
  - Gray: `L1=0.0311`, `bin error=31.72`, `Hamming=0.1649`
- Example at `M=1024`:
  - Natural: `L1=0.0554`, `bin error=28.27`, `Hamming=0.1938`
  - Gray: `L1=0.0304`, `bin error=15.47`, `Hamming=0.1498`
- Dense still has the best offline regression-style metrics overall, so these artifacts support the narrower claim that Gray improves bitwise learnability relative to natural binary, not that bitwise beats Dense on offline imitation error.

Suggested command:

```bash
python experiments/bc_validation_metrics.py
```

## Expected Decision Use

This artifact now supplies the cheapest mechanism-level evidence for the paper:

- rollout success shows Gray helps at high resolution
- offline metrics show Gray improves code-space locality and action prediction quality relative to natural binary

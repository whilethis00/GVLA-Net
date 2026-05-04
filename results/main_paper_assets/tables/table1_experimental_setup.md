# Table 1. Experimental Setup

Main-paper setup summary for the controlled behavior-cloning comparison.

## Sources

- `results/validation_split_lock.md`
- `results/comparison_lock.md`

## Table

| Item | Setting |
| --- | --- |
| Environment | robosuite / RoboMimic low-dimensional manipulation |
| Dataset | RoboMimic Lift PH low-dimensional demonstrations |
| Validation split | 10%, seed 20260503 |
| Action | 7D continuous action, discretized per dimension |
| Policy | Behavior cloning with shared MLP backbone |
| Dense baseline | per-dimension M-way categorical head |
| Bitwise heads | Natural, Gray, Random code |
| Metrics | rollout success, action L1/L2, bin error, Hamming error |

## Notes

- Main comparison rows use the matched BC contract from `experiments/bc_train.py` and `experiments/bc_eval.py`.
- The main paper should not mix these rows with VLA-backbone or synthetic pilot results.

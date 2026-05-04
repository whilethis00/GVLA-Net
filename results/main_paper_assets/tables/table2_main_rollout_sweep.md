# Table 2. Main Rollout Success Sweep

All entries are 50-rollout success rates. The `M=2048` results come from a separate later run. The central comparison is Gray-coded bitwise vs natural-binary bitwise; Dense remains a mixed expressive baseline.

## Sources

- `results/final_submission_results.md`
- `experiments/results/bc_study/eval_results.json`
- `experiments/results/bc_study/eval_results_2048.json`

## Table

| Head | Encoding | M=128 | M=256 | M=1024 | M=2048 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense CE | N/A | 10.0% (5/50) | 16.0% (8/50) | 4.0% (2/50) | 16.0% (8/50) |
| Bitwise | Natural | 2.0% (1/50) | 4.0% (2/50) | 2.0% (1/50) | 4.0% (2/50) |
| Bitwise | Gray | 16.0% (8/50) | 10.0% (5/50) | 18.0% (9/50) | 24.0% (12/50) |

## Notes

- The main claim is `Gray > Natural`, not `Gray > Dense`.
- Do not mix 50-rollout rows with the separate 200-rollout robustness study in this table.

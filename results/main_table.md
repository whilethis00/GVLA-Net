# Main Table

This file locks the BC success table for the NeurIPS 2026 rescoping draft.

Sources:

- `experiments/results/bc_study/eval_results.json`
- `experiments/results/bc_study/eval_results_2048.json`

## Locked 50-Rollout Success Table

| Head | Encoding | M=128 | M=256 | M=1024 | M=2048 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense CE | N/A | 10.0% (5/50) | 16.0% (8/50) | 4.0% (2/50) | 16.0% (8/50) |
| Bitwise | Natural | 2.0% (1/50) | 4.0% (2/50) | 2.0% (1/50) | 4.0% (2/50) |
| Bitwise | Gray | 16.0% (8/50) | 10.0% (5/50) | 18.0% (9/50) | 24.0% (12/50) |

## Provenance Note

- `M=128 / 256 / 1024` come from the saved aggregate BC sweep in `eval_results.json`.
- `M=2048` comes from the separate later artifact `eval_results_2048.json`.
- The `2048` row is usable for success comparison because the saved checkpoints and rollout summaries exist for all three compared heads.

## Caveats

1. `M=2048` is not part of the saved matched latency evidence because that later run used `--skip_latency`.
2. The table is still mostly `single-seed`.
3. For `Gray, M=128`, keep the strict saved main-table value `8/50 = 16.0%`.

## Confidence Note

`EXPERIMENTS.md` separately reports a higher-confidence re-evaluation for `Gray, M=128`:

- `31/200 = 15.5%`

Use that only as an appendix or narrative confirmation. Do not mix it into the main row.

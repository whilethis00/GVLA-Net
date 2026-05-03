# Working Status For Codex

## Purpose

This is the handoff note for the current NeurIPS 2026 rescoping push.

It consolidates the latest artifact findings so the remaining writing actions are obvious:

1. use the locked main table and evidence notes consistently
2. stop treating validation metrics or matched end-to-end latency as missing
3. focus remaining effort on paper wording and claim discipline

## Main Table Status

Main BC comparison is now confirmed for:

- `Dense CE`
- `Bitwise / Natural`
- `Bitwise / Gray`

at:

- `M=128`
- `M=256`
- `M=1024`
- `M=2048`

Primary saved artifact files:

- [eval_results.json](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/eval_results.json:1)
- [eval_results_2048.json](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/eval_results_2048.json:1)

Checkpoint roots:

- `experiments/results/bc_study/checkpoints/dense_*`
- `experiments/results/bc_study/checkpoints/gvla_*`
- `experiments/results/bc_study/checkpoints/gvla_gray_*`

Confirmed result provenance:

- `128 / 256 / 1024` come from the saved aggregate BC study sweep
- `2048` exists as a separate later run but appears comparable on training and evaluation settings

## Main Table Numbers To Use

### From `eval_results.json`

| Head | Encoding | M=128 | M=256 | M=1024 |
| --- | --- | ---: | ---: | ---: |
| Dense CE | N/A | 10.0% (5/50) | 16.0% (8/50) | 4.0% (2/50) |
| Bitwise | Natural | 2.0% (1/50) | 4.0% (2/50) | 2.0% (1/50) |
| Bitwise | Gray | 16.0% (8/50) | 10.0% (5/50) | 18.0% (9/50) |

### From `eval_results_2048.json`

Integrated into the final main table:

- Dense CE: `16.0% (8/50)`
- Bitwise / Natural: `4.0% (2/50)`
- Bitwise / Gray: `24.0% (12/50)`

## Main Table Caveats

1. `M=2048` was run separately via [run_bc_gray_2048.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/run_bc_gray_2048.sh:1), not in the original `8..1024` sweep.
2. `2048` is comparable for rollout success, but not for latency provenance because that run used `--skip_latency`.
3. BC comparison is still mostly `single-seed`.
4. Do not mix inconsistent high-confidence Gray re-evals into the main row:
   - multiple `200-rollout` Gray-128 artifacts disagree
   - keep the main row on the strict saved `50-rollout` table unless explicitly moved to appendix discussion

## What Actually Exists For Validation Metrics

Saved validation artifacts now exist at:

- [validation_metrics.csv](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/validation_metrics/validation_metrics.csv:1)
- [validation_metrics.md](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/validation_metrics/validation_metrics.md:1)
- [validation_split_lock.json](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/validation_metrics/validation_split_lock.json:1)

Saved takeaway:

- Gray beats natural binary on action `L1`, action `L2`, mean bin error, and mean Hamming error at every tested `M`.
- At `M=2048`, natural has `L1=0.0551`, `bin error=56.37`, `Hamming=0.2030`.
- At `M=2048`, Gray has `L1=0.0311`, `bin error=31.72`, `Hamming=0.1649`.
- Dense still has the best offline imitation metrics overall, so the mechanism claim must stay narrow: Gray improves bitwise learnability relative to natural binary.

## Metrics Included

The saved validation artifact includes, for each main-table row:

- action `L1`
- action `L2`
- bin error `|b_hat - b|`
- Hamming error
- adjacent-bin error rate

Dataset / split note:

- the shared validation split is now frozen in [validation_split_lock.json](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/validation_metrics/validation_split_lock.json:1)

## What Actually Exists For Latency

Existing head-only latency artifacts:

- [experiments/results/bc_study/latency_batch.json](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/latency_batch.json:1)
- [experiments/bc_latency_batch.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/bc_latency_batch.py:1)
- [experiments/bc_eval.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/bc_eval.py:159)

Missing:
- none for the locked BC evidence package

Matched end-to-end artifact now exists at:

- [end_to_end_latency.json](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/end_to_end_latency/end_to_end_latency.json:1)
- [end_to_end_latency.md](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/bc_study/end_to_end_latency/end_to_end_latency.md:1)

Saved takeaway:

- At small `M` and `batch=1`, Dense remains competitive or faster.
- At `M=2048, batch=1`, Dense is `1.5967 ms`, Natural is `0.8061 ms`, Gray is `1.2919 ms`.
- At `M=2048, batch=256`, Dense is `7.6192 ms`, Natural is `1.9229 ms`, Gray is `2.8445 ms`.
- Gray is somewhat slower than natural within the bitwise family, so success and latency should be discussed as separate axes.

## Immediate Next Actions

### P0

Use the locked evidence consistently in the paper text:

- main success table through `M=2048`
- validation metrics as mechanism evidence for `Gray > Natural`
- end-to-end latency as a qualified high-resolution efficiency result

### P1

Keep claims disciplined:

- do not say `bitwise beats Dense on offline imitation error`
- do not say `Gray is the fastest bitwise variant`
- do say `Gray improves learnability relative to natural binary`
- do say `bitwise becomes latency-favorable in the high-resolution regime`

## Things Not To Do

- do not start new training unless an artifact is actually missing
- do not spend time reconciling every Gray 200-rollout inconsistency for the main table
- do not expand scope to VLA / Octo / image observations
- do not overclaim from head-only latency

## Bottom Line

The paper now has the core locked evidence package:

1. main success table through `M=2048`
2. offline validation metrics from existing checkpoints
3. matched end-to-end latency from existing checkpoints

The remaining work is no longer artifact recovery. It is disciplined interpretation and paper writing.

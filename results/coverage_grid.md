# Coverage Grid

## Purpose

This file tracks which result cells already exist for the NeurIPS 2026 rescoping submission and which ones are still missing.

Main question:

> Do we already have a clean, comparable result for each method and each resolution needed in the main paper?

Use this file before launching any new evaluation or training.

## Comparison Lock

All main-table entries must satisfy the same comparison contract.

- shared backbone family: `low-dimensional BC policy from experiments/bc_train.py`
- shared dataset / split: `RoboMimic Lift (PH demos), evaluated in robosuite Lift / Panda`
- shared action dimension: `shared across compared runs; latency scripts use action_dim=7`
- shared quantization range: `must be inherited from the same bc_train.py quantization setup; not yet explicitly locked in an artifact note`
- shared evaluation protocol: `epochs=200 training, rollout eval with n_rollouts=50 for main saved table`
- shared success definition: `Lift task success from experiments/bc_eval.py / robosuite Lift evaluation`
- shared checkpoint selection rule: `best checkpoint / saved eval result as recorded in eval_results.json; exact provenance file still missing`

If any run violates one of the items above, mark it as `appendix-only` or `reject`.

Current caveats to keep visible:

- main BC comparison is `single-seed` for most saved results
- `M=128` Gray has a separate higher-confidence `200-rollout` re-eval note in `EXPERIMENTS.md`
- `M=2048` success artifacts exist, but they come from a separate later run and do not carry matched saved latency

## Main Table Coverage

Legend:

- `yes`: checkpoint and evaluation both exist and are comparable
- `eval-only`: checkpoint exists, evaluation still needs to be run
- `ckpt-only`: result summary exists but provenance is incomplete
- `missing`: no usable artifact found yet
- `appendix-only`: exists but not comparable enough for the main table
- `reject`: should not be used

| Method | Encoding | M=128 | M=256 | M=1024 | M=2048 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Dense CE | N/A | `yes` | `yes` | `yes` | `yes` | `128: 5/50 = 10%; 256: 8/50 = 16%; 1024: 2/50 = 4% from eval_results.json; 2048: 8/50 = 16% from eval_results_2048.json` |
| Bitwise | Natural | `yes` | `yes` | `yes` | `yes` | `128: 1/50 = 2%; 256: 2/50 = 4%; 1024: 1/50 = 2% from eval_results.json; 2048: 2/50 = 4% from eval_results_2048.json` |
| Bitwise | Gray | `yes` | `yes` | `yes` | `yes` | `128: eval_results has 8/50 = 16%, while EXPERIMENTS.md notes a separate 31/200 = 15.5% re-eval; 256: 5/50 = 10%; 1024: 9/50 = 18%; 2048: 12/50 = 24% from eval_results_2048.json` |

## Artifact Registry

Fill one row per candidate run.

| Method | Encoding | M | Checkpoint Path | Eval Log Path | Success Summary Path | Comparable? | Status | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `Dense CE` | `N/A` | `128` | `experiments/results/bc_study/checkpoints/dense_128/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Dense CE` | `N/A` | `256` | `experiments/results/bc_study/checkpoints/dense_256/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Dense CE` | `N/A` | `1024` | `experiments/results/bc_study/checkpoints/dense_1024/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Dense CE` | `N/A` | `2048` | `experiments/results/bc_study/checkpoints/dense_2048/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results_2048.json` | `yes` | `yes` | `separate later run; success comparable, latency not saved` |
| `Bitwise` | `Natural` | `128` | `experiments/results/bc_study/checkpoints/gvla_128/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Bitwise` | `Natural` | `256` | `experiments/results/bc_study/checkpoints/gvla_256/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Bitwise` | `Natural` | `1024` | `experiments/results/bc_study/checkpoints/gvla_1024/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `resume helper also exists at experiments/resume_gvla_1024_natural_eval.py` |
| `Bitwise` | `Natural` | `2048` | `experiments/results/bc_study/checkpoints/gvla_2048/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results_2048.json` | `yes` | `yes` | `separate later run; success comparable, latency not saved` |
| `Bitwise` | `Gray` | `128` | `experiments/results/bc_study/checkpoints/gvla_gray_128/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `main row stays on 8/50; 31/200 note remains appendix-only` |
| `Bitwise` | `Gray` | `256` | `experiments/results/bc_study/checkpoints/gvla_gray_256/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Bitwise` | `Gray` | `1024` | `experiments/results/bc_study/checkpoints/gvla_gray_1024/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results.json` | `yes` | `yes` | `config.json also present` |
| `Bitwise` | `Gray` | `2048` | `experiments/results/bc_study/checkpoints/gvla_gray_2048/best.pt` | `not separately saved` | `experiments/results/bc_study/eval_results_2048.json` | `yes` | `yes` | `separate later run; success comparable, latency not saved` |

## Validation Metrics Coverage

These are required to support the mechanism claim even if rollout success is noisy.

| Method | Encoding | M | Predictions Saved? | GT Validation Set Locked? | Bin Metadata Locked? | Bit Outputs Available? | Ready For Metrics? | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `Dense / Natural / Gray` | `mixed` | `128/256/1024/2048` | `yes` | `yes` | `yes` | `yes` | `yes` | `validation_metrics.{csv,json,md}` and per-run prediction CSVs now exist under experiments/results/bc_study/validation_metrics/` |

Required metric outputs:

- action `L1`
- action `L2`
- bin error `|b_hat - b|`
- Hamming error
- adjacent-bin error rate

## Latency Coverage

Track whether head-only and end-to-end timing can be produced cleanly.

| Method | Encoding | M | Head-Only Script Ready? | E2E Script Ready? | Hardware Matched? | Batch=1 Measured? | Large Batch Measured? | Status | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `Dense vs GVLA` | `matched BC end-to-end` | `128/256/1024/2048` | `yes` | `yes` | `cpu matched within the saved artifact` | `yes` | `yes` | `yes` | `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.json` now records full BCPolicy.predict timing for Dense, Natural, and Gray at batch=1 and 256` |
| `Dense vs GVLA` | `head-only` | `8..65536` | `yes` | `yes` | `yes for saved head-only BC study` | `yes` | `yes` | `yes` | `experiments/results/bc_study/latency_batch.json` covers batch x M head-only; use together with the matched end-to-end artifact, not as a substitute for it` |
| `Dense vs GVLA` | `head-only synthetic / robosuite` | `1024..4194304 total N` | `yes` | `no` | `cpu for robosuite_study results` | `n/a` | `n/a` | `appendix-only` | `experiments/results/robosuite_study/results.json contains latency only, but this is not the matched BC main comparison` |

Timing protocol fields to lock:

- device / GPU model: `A100 for bc_study head-only sweep; cpu appears in robosuite_study synthetic latency file`
- framework version: `torch 1.12.1 recorded in end_to_end_latency.json`
- warm-up iterations: `100 recorded in end_to_end_latency.json`
- timed iterations: `1000 recorded in end_to_end_latency.json`
- synchronization method: `cpu timing path recorded in bc_end_to_end_latency.py`
- statistic reported: `mean latency per forward pass`

## Missing Cells Summary

List only the cells that block the paper.

| Priority | Artifact | Method | Encoding | M | Missing Piece | Cheapest Fix | Requires GPU? | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| `P1` | `main-table` | `Bitwise Gray` | `Gray` | `128` | `separate 200-rollout note handling` | `keep 8/50 in main row and move 31/200 to appendix text only` | `no` | `no longer blocks the locked table` |
| `P2` | `latency` | `robosuite synthetic` | `n/a` | `huge N` | `comparison lock` | `appendix only unless needed for qualitative scaling context` | `no` | `not the matched BC evidence set` |

Priority rule:

- `P0`: blocks the main paper claim
- `P1`: strongly improves evidence but not strictly blocking
- `P2`: appendix or bonus only

## Decisions

Record binding decisions here so the comparison set stops drifting.

- `[2026-05-03]` main-table methods locked to: `Dense CE / Bitwise Natural / Bitwise Gray`
- `[2026-05-03]` allowed `M` values locked to: `128 / 256 / 1024 / 2048 for success, validation metrics, and matched end-to-end latency`
- `[2026-05-03]` latency evidence locked to: `head-only latency_batch.json plus matched end_to_end_latency.json`
- `[2026-05-03]` validation metrics locked to: `validation_metrics.{csv,json,md} plus validation_split_lock.json`
- `[date]` appendix-only runs: `robosuite_study latency-only artifacts; any fast-mode CPU BC results; Gray 128 31/200 re-eval note`
- `[date]` rejected runs: `none yet, but any mismatched quantization-range or split run should be dropped from the main table`

## Final Gate

Do not start new training until this file answers:

1. Which exact main-table cells are already covered?
2. Which missing cells can be filled by evaluation only?
3. Which missing cells truly require new training?
4. Which existing runs are not comparable and must be excluded?

Current answer:

- no new training is required for the locked main claim
- the remaining work is interpretation and paper writing, not artifact recovery

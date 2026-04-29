# Robosuite AR Comparison Progress

Last updated: 2026-04-29

## Setup

- Script: `experiments/robosuite_quantization_study.py`
- Task: `pick_place_can`
- Rollouts: `50`
- Horizon: `400`
- Device: `cpu`
- Decode mode: `ar_tokens` for AR runs, `gvla` for GVLA sweep
- Notes:
  - The latency model in this script is not measured wall-clock latency.
  - For AR runs, the implementation applies accumulated token delay as stale-action steps:
    `total_decode_ms = token_latency_ms * 7`.
  - Control frequency is `20Hz`, so the control period is `50ms`.

## Completed Results

### GVLA quantisation sweep

Source: `experiments/results/robosuite_ar_compare_gvla/results.json`

| Setting | Success rate |
|---|---:|
| continuous | 100.0% |
| GVLA, 32 bins/dim | 0.0% |
| GVLA, 48 bins/dim | 8.0% |
| GVLA, 64 bins/dim | 84.0% |
| GVLA, 96 bins/dim | 100.0% |
| GVLA, 128 bins/dim | 100.0% |

Takeaway: GVLA is very sensitive to coarse action discretisation on this task; roughly `96+ bins/dim` are needed to recover the continuous baseline.

### AR baseline and mild perturbations

| Setting | Source | Success rate |
|---|---|---:|
| AR 256 bins, 0ms | `robosuite_ar_compare_ar0/results.json` | 100.0% |
| AR 256 bins, 8ms | `robosuite_ar_compare_ar8/results.json` | 100.0% |
| AR 256 bins, 10ms | `robosuite_ar_compare_ar10/results.json` | 100.0% |
| AR 256 bins, 15ms | `robosuite_ar_compare_ar15/results.json` | 100.0% |
| AR 256 bins, 25ms | `robosuite_ar_compare_ar25/results.json` | 100.0% |
| AR 256 bins, error 0.02 | `robosuite_ar_compare_err002/results.json` | 100.0% |
| AR 256 bins, error 0.05 | `robosuite_ar_compare_err005/results.json` | 100.0% |
| AR 256 bins, error 0.10 | `robosuite_ar_compare_err010/results.json` | 100.0% |

Takeaway: under mild latency and token corruption, AR remains fully robust on this task.

### AR high-latency stress

| Setting | Continuous baseline inside same run | AR result |
|---|---:|---:|
| AR 256 bins, 50ms/token | 16.0% | 28.0% |
| AR 256 bins, 75ms/token | 8.0% | 8.0% |

Sources:

- `experiments/results/robosuite_ar_compare_ar50_queue/results.json`
- `experiments/results/robosuite_ar_compare_ar75_queue/results.json`

Takeaway:

- At `50ms/token`, accumulated AR decode delay is large enough to cause a major collapse.
- At `75ms/token`, both the continuous baseline and AR are nearly unusable.
- This supports the claim that sufficiently large decode delay can quickly destroy control performance.

## In Progress

The following queue was launched via `experiments/run_robosuite_ar_queue.sh`:

1. `robosuite_ar_compare_ar50_queue` completed
2. `robosuite_ar_compare_ar75_queue` completed
3. `robosuite_ar_compare_ar100_queue` running or pending final write
4. `robosuite_ar_compare_err020_queue`
5. `robosuite_ar_compare_err030_queue`
6. `robosuite_ar_compare_ar128_queue`
7. `robosuite_ar_compare_ar128_lat50_queue`
8. `robosuite_ar_compare_ar96_queue`
9. `robosuite_ar_compare_ar96_lat50_queue`

## Current Interpretation

- The current data do not support the claim that AR tokenization is weak under ideal or near-ideal settings.
- The current data do support the claim that AR pipelines become brittle once decode delay becomes sufficiently large.
- The strongest story for GVLA is therefore not "AR never works", but:
  "AR can work well in the high-fidelity regime, but preserving that performance requires a favorable latency / token-budget regime. Faster structured action representations become important once that regime is no longer available."

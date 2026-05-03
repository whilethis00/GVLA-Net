# Latency Protocol

## Goal

Produce an honest latency section that distinguishes:

1. head-only latency
2. end-to-end latency

and does not overclaim from one regime to the other.

## Current Status

What exists:

- `experiments/results/bc_study/latency_batch.json`
  - batch x `M` head-only latency
  - includes `batch=1` and large-batch regimes
- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.json`
  - matched `BCPolicy.predict` end-to-end latency for `Dense CE`, `Bitwise / Natural`, and `Bitwise / Gray`
  - includes `M=128 / 256 / 1024 / 2048`
  - includes `batch=1` and `batch=256`
- `experiments/results/robosuite_study/results.json`
  - synthetic or auxiliary latency-only scaling data

What now exists:

- executable wrapper: `experiments/bc_end_to_end_latency.py`

## Locked Interpretation

The paper should say:

> Head-only latency does not always translate to end-to-end latency. In the saved BCPolicy.predict artifact on A100, Dense remains faster than both bitwise variants through `M=2048`, even though separate head-only scaling evidence still favors bitwise heads in larger-`M` / larger-batch regimes.

The paper should not say:

- `GVLA is always faster`
- `head-only speedup implies online control speedup`

## Existing Head-Only Evidence

From `experiments/results/bc_study/latency_batch.json`:

- Dense is faster at small `M` / batch `1`
- Dense cost rises sharply at large `M` and large batch
- GVLA remains comparatively flat across `M` and batch size in the saved sweep

This is already enough for a careful head-only claim.

## Matched End-to-End Artifact

Measured with `experiments/bc_end_to_end_latency.py`:

- Dense CE
- Bitwise / Natural
- Bitwise / Gray
- shared backbone
- batch `1`
- larger batch `256`
- same timing protocol
- reused saved checkpoints rather than retraining

Key measured regime:

- `M=128, batch=1`: Dense `0.8464 ms`, Natural `2.4915 ms`, Gray `5.0921 ms`
- `M=1024, batch=256`: Dense `0.8597 ms`, Natural `2.4994 ms`, Gray `5.8739 ms`
- `M=2048, batch=1`: Dense `0.8396 ms`, Natural `2.4855 ms`, Gray `6.4359 ms`
- `M=2048, batch=256`: Dense `0.8551 ms`, Natural `2.5011 ms`, Gray `6.1949 ms`

Interpretation to keep:

- Dense is still faster at small `M` and `batch=1`.
- In this matched BC artifact, Dense stays faster than Natural and Gray through `M=2048` for both `batch=1` and `batch=256`.
- Natural bitwise is the faster bitwise variant, but not faster than Dense in the measured BC regime.
- Gray improves learning quality relative to natural binary, but is somewhat slower than natural within the bitwise family.

## Protocol Fields To Record

- device / GPU model
- framework version
- precision mode: default PyTorch eager inference
- warm-up iterations
- timed iterations
- synchronization method
- statistic reported: `mean`

Recorded for the saved artifact:

- device: `cuda`
- GPU model: `NVIDIA A100-SXM4-80GB`
- torch: `2.6.0+cu124`
- warm-up iterations: `100`
- timed iterations: `1000`
- synchronization method: `torch.cuda.Event` timing + CUDA synchronize
- statistic reported: arithmetic mean latency per forward pass

## Minimum Deliverables

- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.csv`
- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.json`
- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.md`

Suggested command:

```bash
python experiments/bc_end_to_end_latency.py
```

## Locked Claim Boundary

- do not say `GVLA is always faster`
- do not collapse Natural and Gray into one latency number
- do say that head-only scaling and end-to-end BC latency answer different questions, and both should be reported separately

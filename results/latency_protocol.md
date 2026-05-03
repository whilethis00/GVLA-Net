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

> Head-only latency does not always translate to end-to-end latency. At small `M` and batch size one, Dense remains competitive because backbone cost and fixed overhead dominate. As `M` grows, Dense end-to-end cost rises while bitwise heads remain flatter; by `M=2048`, bitwise natural is faster than Dense at both `batch=1` and `batch=256`.

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

- `M=128, batch=1`: Dense `0.3649 ms`, Natural `0.8031 ms`, Gray `1.1246 ms`
- `M=1024, batch=256`: Dense `4.1445 ms`, Natural `1.8773 ms`, Gray `2.7704 ms`
- `M=2048, batch=1`: Dense `1.5967 ms`, Natural `0.8061 ms`, Gray `1.2919 ms`
- `M=2048, batch=256`: Dense `7.6192 ms`, Natural `1.9229 ms`, Gray `2.8445 ms`

Interpretation to keep:

- Dense is still faster at small `M` and `batch=1`.
- Natural bitwise overtakes Dense by `M=2048` even at `batch=1`.
- Both bitwise heads beat Dense clearly at larger batch in the high-resolution regime.
- Gray improves learning quality relative to natural binary, but is somewhat slower than natural within the bitwise family.

## Protocol Fields To Record

- device / GPU model: `cpu` in the saved matched end-to-end artifact
- framework version
- precision mode: default PyTorch eager inference
- warm-up iterations
- timed iterations
- synchronization method
- statistic reported: `mean`

Recorded for the saved artifact:

- device: `cpu`
- torch: `1.12.1`
- warm-up iterations: `100`
- timed iterations: `1000`
- synchronization method: `none needed on cpu`
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
- do say that high-resolution end-to-end latency now supports the claim that Dense becomes unfavorable as `M` grows

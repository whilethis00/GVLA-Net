# Comparison Lock

## Purpose

This file freezes the comparison contract for the NeurIPS 2026 rescoping submission.

Use it to prevent the main table from drifting into a mixture of incompatible runs.

## Locked Main Claim

The paper is not a broad "GVLA solves structured output spaces" claim.

The main empirical claim is narrower:

> In high-resolution discretized robot control, bitwise action heads are sensitive to target-code geometry, and Gray coding materially improves learnability relative to natural binary coding.

## Locked Main Evidence

The main evidence package consists of exactly three pieces:

1. resolution-sensitive success in RoboMimic Lift / robosuite Lift BC
2. Gray vs natural code comparison at matched training settings
3. honest latency analysis with head-only and, if available, matched end-to-end timings

Everything else is appendix-level or should be dropped.

## Locked Comparison Contract

Main-table rows must satisfy all of the following:

- same task family: `RoboMimic Lift` demos evaluated in `robosuite Lift` with `Panda`
- same model family: `low-dimensional BC policy` from `experiments/bc_train.py`
- same training budget: `epochs=200`
- same evaluation protocol: `n_rollouts=50` for the saved base comparison, unless a higher-confidence re-eval is explicitly called out
- same output-head comparison:
  - `Dense CE`
  - `Bitwise / Natural`
  - `Bitwise / Gray`
- same success definition from `experiments/bc_eval.py`

Main-table rows must not mix in:

- CPU fast-mode pilot runs
- synthetic robosuite quantization-study rollouts
- orthogonality ablation runs as if they were baseline rows
- VLA or Octo backbone experiments

## Currently Usable Resolution Set

Confirmed from saved BC artifacts:

- `M=128`
- `M=256`
- `M=1024`
- `M=2048`

Implication:

> The main success table can now span `128 / 256 / 1024 / 2048`, but `2048` must carry a provenance note because it comes from a separate later saved run.

## Known Caveats

1. Most saved BC comparison results are `single-seed`.
2. The Gray `M=128` row has two nearby values in notes:
   - `8/50 = 16%` in `eval_results.json`
   - `31/200 = 15.5%` in `EXPERIMENTS.md`
3. Checkpoint provenance for the saved BC table is described in notes, but the checkpoint directories were not yet directly enumerated in the current artifact sweep.
4. Validation metrics artifacts have not yet been found.
5. End-to-end latency artifacts have not yet been found.
6. `M=2048` success is usable, but it is not yet part of matched latency evidence.

## Binding Decisions

### Decision 1

The main paper should assume:

- primary success table: `M=128, 256, 1024, 2048`
- explicit provenance note: `2048` comes from a separate later saved run
- explicit latency caveat: `2048` is not yet covered by matched saved latency evidence

### Decision 2

For `M=128` Gray, choose one of the following and use it consistently:

- base table uses the saved `50-rollout` result for strict row alignment
- appendix note reports the separate `200-rollout` re-evaluation for confidence

Do not mix `50-rollout` and `200-rollout` numbers inside one main row without explicit labeling.

### Decision 3

Orthogonality is not a main comparison axis.

- keep any orthogonality result in appendix or limitations discussion
- do not let it mutate the Dense / Natural / Gray baseline table

## Remaining Locks To Resolve

- quantization range needs an explicit artifact note
- validation split for the metric dump needs to be frozen
- latency protocol note needs to be written down in one file

## Final Rule

If a run is not cleanly matched to this contract, it does not go into the main table.

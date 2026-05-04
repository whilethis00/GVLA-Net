# Code Geometry Matters for Bitwise Action Prediction in High-Resolution Robot Control

This repository contains code and result artifacts for studying code geometry in bitwise action heads for high-resolution discretized robot control. The main comparison is between dense per-dimension categorical heads, natural-binary bitwise heads, Gray-coded bitwise heads, and random-code controls in a behavior cloning setup.

The central empirical claim is narrow:

> High-resolution discretized robot control can benefit from bitwise output factorization, but the learnability of bitwise heads depends critically on target-code geometry. Gray coding preserves locality and substantially improves bitwise action prediction relative to natural binary coding.

This repository is intended to support controlled behavior-cloning experiments. It should not be read as evidence of universal VLA-scale acceleration or as a claim that bitwise heads universally dominate dense categorical heads.

## Scope

Each action dimension is discretized into `M` bins. A dense head predicts `M` logits per dimension. A bitwise head predicts `k = ceil(log2 M)` bits per dimension and decodes the resulting codeword back to a bin index.

An explicit limitation matters here:

> The bitwise head is not distributionally equivalent to a dense categorical head. It trades categorical expressivity for output efficiency by factorizing each action dimension through a binary code.

The paper-facing question is therefore not whether bitwise heads solve all large output spaces, but whether the target code geometry changes how learnable bitwise action prediction is under a bit-wise loss.

## Main Findings

- Gray-coded bitwise heads are substantially more learnable than natural-binary bitwise heads in the locked behavior-cloning setting.
- The Gray-vs-Natural gap appears in validation metrics, rollout success, and code-geometry diagnostics.
- Random code assignments collapse badly, supporting the claim that supervision geometry matters.
- Removing the orthogonality regularizer does not remove the main Gray-vs-Natural effect in the current setup.
- Matched end-to-end latency does not beat the dense baseline up to `M=2048` in the current BC setup, so latency is a qualified secondary analysis rather than the main result.

Representative locked numbers:

- Validation at `M=1024`: Natural `L1=0.0554`, `Bin Error=28.2745`, `Hamming=0.1938`; Gray `L1=0.0304`, `Bin Error=15.4692`, `Hamming=0.1498`.
- Code-geometry diagnostic: consecutive target bit flips are Natural `2.7796`, Gray `2.2483`, Random `4.0069`.
- Robustness at `M=1024`: Natural `7/200 = 3.5%`, Gray `42/200 = 21.0%`.
- Robustness at `M=2048`: Natural `4/200 = 2.0%`, Gray `23/200 = 11.5%`.

## Repository Layout

- `models/layers.py`: shared model layers used by the BC experiments.
- `experiments/bc_train.py`: BC training entry point.
- `experiments/bc_eval.py`: rollout evaluation entry point.
- `experiments/bc_validation_metrics.py`: validation metrics used in the paper tables.
- `experiments/bc_rollout_robustness.py`: 200-rollout robustness evaluation.
- `experiments/bc_end_to_end_latency.py`: matched end-to-end latency measurement.
- `experiments/bc_target_bit_transition_diagnostic.py`: code-geometry diagnostic on target transitions.
- `experiments/results/bc_study/`: locked result artifacts used for the paper.
- `results/final_submission_results.md`: locked submission-facing summary.
- `results/main_vs_appendix_plan.md`: paper assembly plan for main text vs appendix.

## Result Artifacts

The paper-facing artifacts live under `experiments/results/bc_study/`:

- `validation_metrics/validation_metrics.md`
- `reviewer_defense_metrics/validation_metrics.md`
- `rollout_robustness/rollout_robustness.md`
- `code_geometry_diagnostic/code_geometry_diagnostic.md`
- `end_to_end_latency/end_to_end_latency.md`
- `end_to_end_latency_gpu/end_to_end_latency.md`

Supporting submission notes live under `results/`:

- `comparison_lock.md`
- `validation_split_lock.md`
- `latency_protocol.md`
- `coverage_grid.md`
- `final_submission_results.md`
- `main_vs_appendix_plan.md`

## Reproduction Notes

This repository currently reflects an in-progress research workspace rather than a minimal supplementary package. For submission-facing reproduction, the core paper results are the locked artifacts listed above plus the BC experiment scripts under `experiments/`.

At a high level, the paper tables come from:

- training checkpoints produced by `experiments/bc_train.py`
- rollout success numbers from `experiments/bc_eval.py` and `experiments/bc_rollout_robustness.py`
- validation tables from `experiments/bc_validation_metrics.py`
- code-geometry diagnostics from `experiments/bc_target_bit_transition_diagnostic.py`
- latency tables from `experiments/bc_end_to_end_latency.py`

## Claim Boundaries

The following are intentionally not claims of this repository or the paper:

- bitwise heads universally beat dense categorical heads
- Gray coding is the fastest variant
- the method solves general large structured output spaces
- the current experiments constitute a VLA-scale deployment result
- matched end-to-end latency is better than dense in the present BC setup

## Limitations

- The main evidence is from a controlled low-dimensional behavior-cloning setting.
- The experiments focus on a narrow task family rather than broad cross-domain generalization.
- Independent bit prediction reduces output size but also reduces expressivity relative to dense categorical prediction.
- Multimodal action distributions may be poorly matched to the factorized bitwise formulation.
- The current matched end-to-end latency results do not show a speedup over dense heads up to `M=2048`.

## Submission Note

If this codebase is packaged for anonymous supplementary material, the paper-facing release should be a trimmed subset centered on the BC experiments and locked results above. Public-repo language about broad VLA acceleration, logarithmic routing, or orthogonality as the primary story should not be used for the final submission package.

# Final Main vs Appendix Plan

This document fixes the final paper framing around a narrow, defensible claim. The paper should be submitted, but only if every section stays within the locked evidence and avoids the older overclaiming story.

## Final Title

> Code Geometry Matters for Bitwise Action Prediction in High-Resolution Robot Control

`GVLA` should remain a method name, not the title framing.

## One-Sentence Claim

> High-resolution discretized robot control can benefit from bitwise output factorization, but the learnability of bitwise heads depends critically on target-code geometry. Gray coding preserves locality and substantially improves bitwise action prediction relative to natural binary coding.

This is the paper's boundary. Every abstract sentence, figure caption, README sentence, and limitation section should remain inside it.

## Claims We Must Not Make

- `GVLA solves large structured output spaces`
- `GVLA is always faster than Dense`
- `Bitwise beats Dense on offline imitation error`
- `Gray is the fastest bitwise variant`
- `This is a VLA-scale result`

Latency is not the main selling point. In the locked matched end-to-end BC setting, Dense remains faster up to `M=2048`, so latency is a qualified secondary analysis only.

## Final Contributions

### 1. Bitwise action head formulation

For `D` action dimensions and `M` bins per dimension:

- Dense predicts `D x M` logits.
- Bitwise predicts `D x k` logits where `k = ceil(log2 M)`.

Required sentence:

> GVLA is not distributionally equivalent to a dense categorical head. It trades categorical expressivity for output efficiency by factorizing each action dimension through a binary code.

### 2. Code geometry as supervision geometry

Natural binary can map neighboring bins to distant codewords, for example `3 = 011` and `4 = 100`. Gray coding preserves local adjacency.

Required sentence:

> Under bit-wise BCE, the target code induces the supervision geometry.

This is the main technical point. The paper is not about inventing binary codes; it is about showing that code geometry controls learnability in this control setting.

### 3. Controlled empirical evidence

The strongest evidence bundle consists of four parts:

1. Validation metrics
   Natural at `M=1024`: `L1=0.0554`, `Bin Error=28.2745`, `Hamming=0.1938`
   Gray at `M=1024`: `L1=0.0304`, `Bin Error=15.4692`, `Hamming=0.1498`
2. Reviewer-defense controls
   `Natural seed2` stays close to Natural.
   `Gray no-orth` stays close to Gray.
   `Random` collapses: `L1=0.4020`, `Bin Error=205.7474`.
3. Code-geometry diagnostic
   Mean target bit flips: Natural `2.7796`, Gray `2.2483`, Random `4.0069`
4. 200-rollout robustness
   `M=1024`: Natural `7/200 = 3.5%`, Gray `42/200 = 21.0%`, `p=6.55e-08`
   `M=2048`: Natural `4/200 = 2.0%`, Gray `23/200 = 11.5%`, `p=1.95e-04`

These four pieces together support `Gray > Natural >> Random` and make `code geometry matters` the clean story.

## Main Paper Structure

### Page 1: Introduction

Start with:

> High-resolution discretized action prediction is attractive for precision-sensitive control, but dense per-dimension categorical heads scale linearly with the number of bins. Bitwise heads reduce the number of output logits, but they also change the learning problem. This paper studies how the target code geometry affects the learnability of bitwise action heads.

Do not mention:

- inference wall
- large structured output space as the main claim
- quantum measurement framing
- VLA-scale deployment language

### Page 2: Problem formulation

Keep the notation simple:

- `a in R^D`
- `b_j = q(a_j) in {0, ..., M-1}`
- Dense logits in `R^(D x M)`
- Bitwise logits in `R^(D x k)` with `k = ceil(log2 M)`

Admit the expressivity limitation immediately.

### Page 3: Method

Minimal method:

- `P = W z`, `P in R^(D x k)`
- bitwise BCE over code bits
- optional orthogonality regularizer `||W W^T - I||_F^2`

Orthogonality is not the main contribution. The no-orth result implies it should be described as an optional regularizer, not as the core explanation.

### Page 4: Code geometry

This is the paper's center.

Figure 1 should combine:

- Natural discontinuity example `3 = 011`, `4 = 100`
- Gray adjacency intuition
- real trajectory diagnostic with bit flips:
  Natural `2.7796`, Gray `2.2483`, Random `4.0069`

### Pages 5-7: Experiments

Only three main questions:

1. Does code geometry affect bitwise action prediction?
2. Is the effect robust?
3. What about efficiency?

Main figures and tables:

1. Main success table
   Dense / Natural / Gray at `M = 128 / 256 / 1024 / 2048`
2. 200-rollout robustness
   Natural vs Gray at `M = 1024 / 2048`
3. Reviewer-defense validation table
   Natural, Natural seed2, Gray, Gray no-orth, Random at `M=1024`

Everything else goes to the appendix.

### Page 8: Related work

Must cite and position against:

- output coding / ECOC
- hierarchical softmax
- binary label encoding
- ordinal regression / thermometer encoding
- discretized continuous control
- diffusion / mixture / flow policies for multimodal control

Required positioning sentence:

> We do not claim to invent binary output codes. Our contribution is to show that, in high-resolution discretized robot control, the code geometry can dominate the learnability of bitwise action heads.

### Page 9: Limitations

State plainly:

- low-dimensional BC setting
- single task family
- bitwise head is less expressive than dense categorical prediction
- multimodal action distributions may break independent bit prediction
- matched end-to-end latency does not improve over Dense in the current BC setup
- this is not a VLA-scale deployment result

These limitations are part of the defense, not an embarrassment.

## Abstract Draft

Use this framing with only minor line-editing:

> High-resolution action discretization can be useful for precision-sensitive robotic control, but standard per-dimension categorical heads scale linearly with the number of bins. We study bitwise action heads as a simple output factorization: each `M`-way action prediction is replaced by `k=ceil(log2 M)` bit predictions. This factorization reduces output dimensionality, but it also changes the learning problem because the target code induces a geometry under bit-wise losses. Natural binary codes can map neighboring action bins to distant codewords, creating discontinuous supervision, while Gray codes preserve local adjacency by ensuring neighboring bins differ by one bit. In controlled behavior cloning experiments with a shared backbone, Gray-coded bitwise heads substantially improve over natural binary bitwise heads. Across 200 evaluation rollouts, Gray improves success from `3.5%` to `21.0%` at `M=1024` and from `2.0%` to `11.5%` at `M=2048`. Validation metrics show the same trend, with Gray reducing action error, bin error, and Hamming error. Additional controls show that the effect is not explained by a second natural-binary seed, does not disappear without orthogonality regularization, and degrades sharply under random code assignments. Matched end-to-end latency does not improve over dense heads in our current BC setup, so we treat efficiency as a qualified secondary analysis. These results frame high-resolution discretized control as an output-factorization problem where code geometry, not only output size, determines learnability.

## Main vs Appendix Allocation

### Keep in main

- code geometry intuition
- main success table
- reviewer-defense validation evidence
- 200-rollout robustness
- explicit claim boundary
- explicit limitation that latency is secondary

### Move to appendix

- full validation tables across all metrics
- full latency tables for CPU and GPU
- protocol details and split lock
- implementation details
- extra ablations and metric definitions

### Keep out of the paper

- quantum or collapse narrative
- large memory-reduction marketing
- VLA-scale wording
- claims that Dense is consistently beaten
- claims that orthogonality is essential

## Appendix Layout

- `A. Experimental protocol`
  dataset, split seed, validation fraction, rollout counts
- `B. Full validation metrics`
  all `128 / 256 / 1024 / 2048` tables
- `C. Rollout robustness`
  200 rollouts, Wilson CI, Fisher exact tests
- `D. Latency`
  CPU and GPU matched end-to-end plus head-only context
- `E. Code geometry diagnostic`
  target bit flips and random-code comparison
- `F. Implementation details`
  quantization, Gray encode/decode, no-orth, random seed
- `G. Limitations`
  expanded version of the main text limitations

## Repository Packaging Guidance

The current public-repo style should not be used as the submission-facing artifact. For paper release or supplementary packaging:

- keep BC experiment code and locked result artifacts
- remove or archive older VLA-scale and quantum-style narratives
- remove claims that orthogonality is essential
- avoid README language about logarithmic routing breakthroughs
- keep the README claim-limited and reproducibility-focused

Recommended submission-facing README opener:

> This repository contains anonymized code for studying code geometry in bitwise action heads for high-resolution discretized robot control. The main comparison is between dense per-dimension categorical heads, natural-binary bitwise heads, Gray-coded bitwise heads, and random-code controls in a behavior cloning setup. The repository is intended to reproduce the paper's controlled experiments, not to claim universal VLA-scale acceleration.

## Final Decision

Submit.

The last work is not more experimentation. It is claim compression. If the paper looks like a careful small paper that proves one precise fact, it can fight for borderline acceptance. If it looks like a breakthrough pitch, it will get punished.

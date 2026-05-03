# Final Submission Results

이 문서는 제출용으로 잠근 최종 결과 요약이다.

논문의 방어 가능한 핵심 주장은 다음 세 가지다.

1. high-resolution discretized control에서는 action resolution과 code geometry가 실제 성능에 중요하다.
2. Gray-coded bitwise head는 Natural binary bitwise head보다 일관되게 더 학습 가능하다.
3. head-only scaling은 bitwise factorization의 효율 잠재력을 보여주지만, matched end-to-end latency에서는 현재 BC setup 기준 `M<=2048`에서 Dense가 더 빠르다.

## 1. Main Success Table

Source:

- `experiments/results/bc_study/eval_results.json`
- `experiments/results/bc_study/eval_results_2048.json`

| Head | Encoding | M=128 | M=256 | M=1024 | M=2048 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense CE | N/A | 10.0% (5/50) | 16.0% (8/50) | 4.0% (2/50) | 16.0% (8/50) |
| Bitwise | Natural | 2.0% (1/50) | 4.0% (2/50) | 2.0% (1/50) | 4.0% (2/50) |
| Bitwise | Gray | 16.0% (8/50) | 10.0% (5/50) | 18.0% (9/50) | 24.0% (12/50) |

해석:

- low resolution에서는 Dense가 경쟁적이다.
- high resolution에서는 Gray-coded bitwise head가 Natural binary보다 일관되게 낫다.
- 특히 `M=1024`와 `M=2048`에서 Gray가 가장 높은 success를 보인다.

주의:

- `M=128 / 256 / 1024`는 aggregate BC sweep에서 왔다.
- `M=2048`는 separate later run에서 왔다.
- main row는 모두 `50-rollout` 기준으로 잠근다.
- Gray `M=128`의 별도 `200-rollout` 수치는 appendix note로만 쓴다.

## 2. Validation Metrics

Source:

- `experiments/results/bc_study/validation_metrics/validation_metrics.md`
- `experiments/results/bc_study/validation_metrics/validation_split_lock.json`

validation split:

- dataset: `data/robomimic/lift/ph/low_dim_v141.hdf5`
- split seed: `20260503`
- val fraction: `0.10`

### Key Rows

| Run | Action L1 | Action L2 | Bin Error | Hamming | Exact Bin Match |
| --- | ---: | ---: | ---: | ---: | ---: |
| gvla_128 | 0.0539 | 0.2486 | 3.3024 | 0.1367 | 0.4797 |
| gvla_gray_128 | 0.0292 | 0.1350 | 1.7016 | 0.0891 | 0.5936 |
| gvla_1024 | 0.0554 | 0.2582 | 28.2745 | 0.1938 | 0.2393 |
| gvla_gray_1024 | 0.0304 | 0.1488 | 15.4692 | 0.1498 | 0.2946 |
| gvla_2048 | 0.0551 | 0.2587 | 56.3696 | 0.2030 | 0.2105 |
| gvla_gray_2048 | 0.0311 | 0.1517 | 31.7193 | 0.1649 | 0.2426 |

핵심 해석:

- Gray는 모든 tested `M`에서 Natural보다 `L1`, `L2`, `bin error`, `Hamming error`가 더 좋다.
- 따라서 main claim은 `Gray improves bitwise learnability relative to Natural`로 잠그는 것이 맞다.
- Dense는 offline imitation metric 자체는 더 좋으므로, `bitwise beats Dense on offline error` 같은 주장은 하지 않는다.

## 3. Reviewer-Defense Additions At `M=1024`

Source:

- `experiments/results/bc_study/reviewer_defense_metrics/validation_metrics.md`

| Run | Encoding | Action L1 | Action L2 | Bin Error | Hamming | Exact Bin Match |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gvla_1024 | Natural | 0.0554 | 0.2582 | 28.2745 | 0.1938 | 0.2393 |
| gvla_1024_seed2 | Natural | 0.0530 | 0.2488 | 27.0196 | 0.1890 | 0.2408 |
| gvla_gray_1024 | Gray | 0.0304 | 0.1488 | 15.4692 | 0.1498 | 0.2946 |
| gvla_gray_1024_noorth | Gray, no-orth | 0.0294 | 0.1429 | 14.9637 | 0.1497 | 0.2934 |
| gvla_random_1024_seed7 | Random | 0.4020 | 1.3917 | 205.7474 | 0.2840 | 0.1819 |

핵심 해석:

- `Natural seed2`가 기존 Natural과 거의 같은 수치를 보여, Gray-vs-Natural 차이가 single-seed artifact일 가능성을 낮춘다.
- `Gray no-orth`가 기존 Gray와 거의 같거나 약간 더 좋아, main effect가 orthogonality regularization이 아니라 code geometry라는 점을 지지한다.
- `Random` code는 validation metric에서 크게 무너져, `Gray > Natural >> Random` 구조를 보여준다.

## 4. Code Geometry Diagnostic

Source:

- `experiments/results/bc_study/code_geometry_diagnostic/code_geometry_diagnostic.md`

실제 demonstration trajectory에서 consecutive action target의 mean bit flips:

| Encoding | All Steps | Small-Action Steps |
| --- | ---: | ---: |
| Natural | 2.7796 | 2.6630 |
| Gray | 2.2483 | 2.1592 |
| Random | 4.0069 | 3.9680 |

핵심 해석:

- 실제 데이터에서도 Gray는 Natural보다 target-bit transition이 더 smooth하다.
- Random code는 가장 거칠다.
- 따라서 `code geometry matters`를 synthetic 예시만이 아니라 실제 trajectory 통계로도 뒷받침할 수 있다.

## 5. Rollout Robustness

Source:

- `experiments/results/bc_study/rollout_robustness/rollout_robustness.md`

`200-rollout` 결과:

| Run | Successes | Success Rate | Wilson 95% CI |
| --- | ---: | ---: | --- |
| gvla_1024 | 7 | 3.5% | [0.017, 0.070] |
| gvla_gray_1024 | 42 | 21.0% | [0.159, 0.272] |
| dense_1024 | 27 | 13.5% | [0.094, 0.189] |
| gvla_2048 | 4 | 2.0% | [0.008, 0.050] |
| gvla_gray_2048 | 23 | 11.5% | [0.078, 0.167] |
| dense_2048 | 18 | 9.0% | [0.058, 0.138] |

Pairwise Fisher exact test:

- `Natural vs Gray @ 1024`: `p = 6.55e-08`
- `Natural vs Gray @ 2048`: `p = 1.95e-04`
- `Dense vs Gray @ 1024`: `p = 0.063`
- `Dense vs Gray @ 2048`: `p = 0.510`

핵심 해석:

- Gray는 `M=1024`와 `M=2048`에서 `200-rollout` 기준으로도 Natural보다 확실히 낫다.
- Gray가 Dense보다 높아 보이는 구간이 있어도, 현재 evidence로는 `Gray > Dense`를 main claim으로 쓰지 않는다.
- 따라서 main success claim은 `Gray beats Natural robustly in the high-resolution regime`로 제한한다.

## 6. End-to-End Latency

Source:

- `experiments/results/bc_study/end_to_end_latency_gpu/end_to_end_latency.md`
- `experiments/results/bc_study/end_to_end_latency_gpu/end_to_end_latency.json`

protocol:

- device: `cuda`
- GPU: `NVIDIA A100-SXM4-80GB`
- torch: `2.1.1+cu118`
- warmup: `100`
- measure: `1000`
- statistic: mean latency per forward pass

### Key Rows

| Run | Batch | Latency (ms) |
| --- | ---: | ---: |
| dense_128 | 1 | 0.7382 |
| gvla_128 | 1 | 2.1944 |
| gvla_gray_128 | 1 | 4.6284 |
| dense_1024 | 256 | 0.7926 |
| gvla_1024 | 256 | 2.2318 |
| gvla_gray_1024 | 256 | 5.3953 |
| dense_2048 | 1 | 0.7361 |
| gvla_2048 | 1 | 2.2069 |
| gvla_gray_2048 | 1 | 5.9004 |
| dense_2048 | 256 | 0.7909 |
| gvla_2048 | 256 | 2.2364 |
| gvla_gray_2048 | 256 | 5.6997 |

핵심 해석:

- small `M`와 `batch=1`에서는 Dense가 더 빠를 수 있다.
- 이번 matched BC artifact에서는 `M=128~2048` 전 구간에서 Dense가 end-to-end latency 기준으로 더 빠르다.
- Dense와 Natural bitwise는 이 범위에서 `M`과 batch 변화에 거의 흔들리지 않는다.
- Gray는 success/validation metric에서는 Natural보다 낫지만, latency는 bitwise family 내부에서 Natural보다 확실히 느리다.
- 따라서 latency는 main selling point가 아니라, head-only scaling evidence를 보완하는 제한된 secondary analysis로만 사용한다.

따라서 latency claim은 아래처럼 제한한다:

> Bitwise heads do not automatically improve end-to-end latency in the current BC regime. Their latency advantage is supported by head-only scaling evidence, while the matched BCPolicy.predict artifact shows that Dense remains faster for `M<=2048` in this setup.

## 7. Locked Paper Claim

이 결과 패키지로 방어 가능한 논문 주장은 아래다.

> High-resolution discretized robot control can benefit from bitwise output factorization, but the learnability of bitwise heads depends critically on target-code geometry. Gray coding preserves locality and substantially improves bitwise action prediction relative to natural binary coding. We do not claim matched end-to-end latency gains over Dense in the current BC setup.

Reviewer-defense addendum:

> This effect is not explained by a single natural-binary seed, does not disappear when orthogonality regularization is removed, degrades sharply under random code assignments, and remains visible across 200 evaluation rollouts in the high-resolution regime.

하지 말아야 할 주장:

- `GVLA solves large structured output spaces`
- `GVLA is always faster than Dense`
- `bitwise beats Dense on offline imitation error`
- `Gray is the fastest bitwise variant`
- `this is a VLA-scale result`

## 8. Submission Checklist For Results

- main table: locked
- validation metrics: locked
- reviewer-defense metrics: locked
- code-geometry diagnostic: locked
- 200-rollout robustness: locked
- end-to-end latency: locked
- claim boundary: locked
- remaining work: paper writing and presentation only

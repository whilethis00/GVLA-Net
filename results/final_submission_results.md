# Final Submission Results

이 문서는 제출용으로 잠근 최종 결과 요약이다.

논문의 방어 가능한 핵심 주장은 다음 세 가지다.

1. high-resolution discretized control에서는 action resolution이 실제 성능에 중요하다.
2. bitwise head의 학습 가능성은 code geometry에 크게 좌우된다.
3. large-`M` regime에서는 dense head보다 bitwise head가 latency 측면에서 유리해질 수 있다.

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

## 3. Code Geometry Diagnostic

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

## 4. End-to-End Latency

Source:

- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.md`
- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.json`

protocol:

- device: `cuda`
- GPU: `NVIDIA A100-SXM4-80GB`
- torch: `2.6.0+cu124`
- warmup: `100`
- measure: `1000`
- statistic: mean latency per forward pass

### Key Rows

| Run | Batch | Latency (ms) |
| --- | ---: | ---: |
| dense_128 | 1 | 0.8464 |
| gvla_128 | 1 | 2.4915 |
| gvla_gray_128 | 1 | 5.0921 |
| dense_1024 | 256 | 0.8597 |
| gvla_1024 | 256 | 2.4994 |
| gvla_gray_1024 | 256 | 5.8739 |
| dense_2048 | 1 | 0.8396 |
| gvla_2048 | 1 | 2.4855 |
| gvla_gray_2048 | 1 | 6.4359 |
| dense_2048 | 256 | 0.8551 |
| gvla_2048 | 256 | 2.5011 |
| gvla_gray_2048 | 256 | 6.1949 |

핵심 해석:

- small `M`와 `batch=1`에서는 Dense가 더 빠를 수 있다.
- 이번 matched BC artifact에서는 `M=128~2048` 전 구간에서 Dense가 end-to-end latency 기준으로 더 빠르다.
- Dense와 Natural bitwise는 이 범위에서 `M`과 batch 변화에 거의 흔들리지 않는다.
- Gray는 success/validation metric에서는 Natural보다 낫지만, latency는 bitwise family 내부에서 Natural보다 확실히 느리다.

따라서 latency claim은 아래처럼 제한한다:

> Bitwise heads do not automatically improve end-to-end latency in the current BC regime. Their latency advantage is supported by head-only scaling evidence, while the matched BCPolicy.predict artifact shows that Dense remains faster for `M<=2048` in this setup.

## 5. Locked Paper Claim

이 결과 패키지로 방어 가능한 논문 주장은 아래다.

> High-resolution discretized robot control can benefit from bitwise output factorization, but the learnability of bitwise heads depends critically on target-code geometry. Gray coding preserves locality and substantially improves bitwise action prediction relative to natural binary coding.

하지 말아야 할 주장:

- `GVLA solves large structured output spaces`
- `GVLA is always faster than Dense`
- `bitwise beats Dense on offline imitation error`
- `Gray is the fastest bitwise variant`
- `this is a VLA-scale result`

## 6. Submission Checklist For Results

- main table: locked
- validation metrics: locked
- end-to-end latency: locked
- claim boundary: locked
- remaining work: paper writing and presentation only

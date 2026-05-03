# Main vs Appendix Plan

이 문서는 현재 잠긴 실험 결과를 기준으로, NeurIPS 제출본의 main paper와 appendix에 무엇을 넣을지 분리한 계획서다.

기준 원칙:

- main에는 `Gray > Natural` 핵심 claim을 직접 지지하는 결과만 넣는다.
- appendix에는 robustness, protocol, 전체 표, 세부 수치를 넣는다.
- latency는 main에서 과장하지 않는다.

## Main Paper

### 1. Main Claim

main text에서 유지할 핵심 주장:

> High-resolution discretized robot control can benefit from bitwise output factorization, but the learnability of bitwise heads depends critically on target-code geometry. Gray coding preserves locality and substantially improves bitwise action prediction relative to natural binary coding.

main text에서 하지 말아야 할 주장:

- `GVLA solves large structured output spaces`
- `GVLA is always faster than Dense`
- `bitwise beats Dense on offline imitation error`
- `Gray is the fastest bitwise variant`
- `this is a VLA-scale result`

### 2. Main Figures / Tables

#### Figure 1: Code Geometry Motivation

넣을 것:

- Natural binary counterexample: `3=011`, `4=100`
- Gray code의 adjacent preservation
- 실제 trajectory 기반 code geometry diagnostic 요약

핵심 수치:

- Natural bit flips: `2.7796`
- Gray bit flips: `2.2483`
- Random bit flips: `4.0069`

메시지:

- Gray는 실제 데이터에서도 Natural보다 smoother target transitions를 만든다.

#### Table 1: Main Success Table

Source:

- `results/final_submission_results.md`

넣을 표:

| Head | Encoding | M=128 | M=256 | M=1024 | M=2048 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense CE | N/A | 10.0% (5/50) | 16.0% (8/50) | 4.0% (2/50) | 16.0% (8/50) |
| Bitwise | Natural | 2.0% (1/50) | 4.0% (2/50) | 2.0% (1/50) | 4.0% (2/50) |
| Bitwise | Gray | 16.0% (8/50) | 10.0% (5/50) | 18.0% (9/50) | 24.0% (12/50) |

메시지:

- Gray는 high-resolution regime에서 Natural보다 consistently better하다.
- Dense와의 비교는 mixed이므로 main claim은 `Gray > Natural`에 한정한다.

주의:

- `M=2048`는 separate later run이라는 provenance note를 표 주석이나 캡션에 넣는다.

#### Table 2 or Figure 2: Validation / Reviewer-Defense Summary

넣을 것:

- `Natural @ 1024`
- `Natural seed2 @ 1024`
- `Gray @ 1024`
- `Gray no-orth @ 1024`
- `Random @ 1024`

핵심 수치:

- Natural: `L1 0.0554`, `Bin Error 28.2745`, `Hamming 0.1938`
- Natural seed2: `L1 0.0530`, `Bin Error 27.0196`, `Hamming 0.1890`
- Gray: `L1 0.0304`, `Bin Error 15.4692`, `Hamming 0.1498`
- Gray no-orth: `L1 0.0294`, `Bin Error 14.9637`, `Hamming 0.1497`
- Random: `L1 0.4020`, `Bin Error 205.7474`, `Hamming 0.2840`

메시지:

- Gray-vs-Natural gap은 single seed artifact가 아니다.
- main effect는 orthogonality보다 code geometry다.
- Random code collapse는 `code geometry matters`를 강하게 뒷받침한다.

#### Figure 3 or Small Table: 200-Rollout Robustness

넣을 것:

- `gvla_1024`
- `gvla_gray_1024`
- `gvla_2048`
- `gvla_gray_2048`

핵심 수치:

- `gvla_1024`: `7/200 = 3.5%`
- `gvla_gray_1024`: `42/200 = 21.0%`
- `gvla_2048`: `4/200 = 2.0%`
- `gvla_gray_2048`: `23/200 = 11.5%`

핵심 통계:

- `Natural vs Gray @ 1024`: `p = 6.55e-08`
- `Natural vs Gray @ 2048`: `p = 1.95e-04`

메시지:

- Gray > Natural은 `200-rollout` 기준에서도 robust하다.

### 3. Main Text Latency Handling

main에는 제한적으로만 넣는다.

넣을 것:

- head-only scaling evidence가 bitwise factorization의 efficiency potential을 보여준다는 점
- matched end-to-end latency에서는 Dense가 현재 BC setup에서 더 빠르다는 점

추천 문장:

> In matched end-to-end latency measurements, bitwise heads did not outperform the dense baseline in the current BC setup up to `M=2048`, on either CPU or GPU. We therefore treat latency as a qualified secondary analysis rather than a primary result.

main에는 full latency table을 크게 넣지 않는다.

## Appendix

### 1. Full Validation Metrics

넣을 것:

- `128 / 256 / 1024 / 2048` 전체 validation metric 표
- exact bin match 포함
- `adjacent_near_miss_rate`는 정의와 함께 appendix에만

Source:

- `experiments/results/bc_study/validation_metrics/validation_metrics.md`
- `experiments/results/bc_study/reviewer_defense_metrics/validation_metrics.md`

### 2. Full Rollout Robustness

넣을 것:

- `200-rollout` 전체 표
- Wilson 95% CI
- Fisher exact test 전체
- `dense` 비교 포함

Source:

- `experiments/results/bc_study/rollout_robustness/rollout_robustness.md`

### 3. Full Latency Tables

넣을 것:

- CPU end-to-end latency 표
- GPU end-to-end latency 표
- head-only latency artifact 요약

Source:

- `experiments/results/bc_study/end_to_end_latency/end_to_end_latency.md`
- `experiments/results/bc_study/end_to_end_latency_gpu/end_to_end_latency.md`
- `experiments/results/bc_study/latency_batch.json`

appendix 메시지:

- head-only scaling은 efficiency potential을 보여준다
- matched end-to-end latency는 current BC setup에서 Dense가 더 빠르다
- 따라서 universal latency speedup claim은 하지 않는다

### 4. Experimental Protocol

넣을 것:

- validation split lock
- latency protocol
- dataset path
- rollout count
- checkpoint provenance

Source:

- `results/validation_split_lock.md`
- `results/latency_protocol.md`
- `results/comparison_lock.md`
- `results/coverage_grid.md`

### 5. Ablation / Reviewer-Defense Details

넣을 것:

- random code seed
- no-orth setting
- additional seed naming
- exact checkpoint paths

### 6. Limitations

appendix에도 재강조할 것:

- low-dimensional BC only
- single task family
- bitwise head는 Dense categorical과 distributionally equivalent하지 않다
- multimodal action settings에서는 부적절할 수 있다
- latency gain은 matched end-to-end 결과에서 확인되지 않았다

## Final Assembly Checklist

### Main에 반드시 있어야 하는 것

- code geometry intuition
- main success table
- validation/reviewer-defense evidence
- 200-rollout robustness
- explicit claim boundary

### Appendix로 보내야 하는 것

- full metric tables
- full latency tables
- protocol details
- extra implementation details
- ambiguous metrics definition

### Main에서 빼야 하는 것

- `adjacent_bin_error` 원래 이름
- 큰 memory exaggeration
- quantum / collapse narrative
- VLA-scale wording
- Dense를 일관적으로 이겼다는 식의 문장

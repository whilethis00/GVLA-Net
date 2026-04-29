# GVLA-Net: Geometry-Aware Binary Action Routing for Large Discrete Action Spaces

최종 업데이트: 2026-04-30

## Paper Positioning

이 논문은 robotics method paper가 아니라, large structured output modeling paper로 쓴다.  
로봇은 핵심 application이지만, 논문의 중심 문제는 다음과 같다.

- high-resolution discretization은 실제로 필요할 수 있다
- 그러나 dense prediction이나 long sequential decoding은 큰 output space에서 비싸진다
- structured bit-wise prediction은 이 large-output regime를 더 잘 다룰 수 있다
- 이때 code geometry는 단순 구현 디테일이 아니라 핵심 학습 신호 설계다

한 줄로 쓰면:

> High-resolution action spaces can matter in precision-sensitive control, but dense prediction does not scale. GVLA makes such regimes tractable through structured bit-wise prediction and geometry-aware coding.

## Title

고정 제목:

> **GVLA-Net: Geometry-Aware Binary Action Routing for Large Discrete Action Spaces**

이 제목은 다음 세 가지를 모두 드러낸다.

- `GVLA-Net`: 방법 자체의 이름
- `Geometry-Aware`: Gray code와 target-space locality의 중요성
- `Large Discrete Action Spaces`: 논문의 실제 문제 설정

## Central Claim

이 논문의 메인 claim은 다음과 같다.

> In precision-sensitive regimes, large action resolution can be necessary for control performance, but conventional dense or sequential output heads scale poorly in that regime. GVLA replaces per-dimension M-way prediction with structured bit-wise prediction, and Gray coding aligns code-space locality with nearby actions, improving high-resolution learning.

이 claim은 세 부분으로 나뉜다.

1. high-resolution regime 자체가 실제로 필요할 수 있다
2. 기존 head는 그 regime에서 비싸다
3. structured bit-wise prediction과 geometry-aware coding이 그 regime에 더 적합하다

## Scope of the Paper

이 논문은 다음을 주장한다.

1. coarse discretization can substantially degrade manipulation success
2. some precision-sensitive regimes require substantially larger action resolution than easy benchmarks
3. dense per-dimension M-way prediction scales poorly with output-space size
4. structured bit-wise action routing offers more favorable scaling
5. Gray coding improves high-resolution GVLA by preserving locality in code space
6. the advantage extends beyond compute to latency and head-memory / VRAM feasibility

이 논문은 다음을 주장하지 않는다.

1. GVLA가 continuous control policy를 전반적으로 이긴다
2. GVLA가 AR tokenization보다 항상 더 정확하다
3. 모든 조작 태스크가 `1024+` resolution을 필요로 한다
4. Gray code가 모든 structured output problem의 최적 code다
5. 현재 결과만으로 humanoid / ALOHA까지 일반화된다

## Abstract-Level Story

초록은 다음 흐름을 가져야 한다.

1. large structured output spaces are expensive to model densely
2. robotics action prediction is a concrete and demanding instance of this problem
3. precision-sensitive control can require substantially larger action resolution than standard easy benchmarks suggest
4. dense or sequential output heads become expensive in this regime
5. GVLA predicts logarithmic-length structured binary codes instead of dense `M`-way outputs
6. Gray coding preserves locality and improves high-resolution learning
7. experiments show quantization sensitivity, high-resolution recovery, coding ablation, and scaling / memory advantages

## Introduction Outline

## Paragraph 1: Large structured output spaces are expensive

도입은 로봇보다 위에서 시작한다.

- many decision problems involve very large structured discrete output spaces
- naively scoring all outputs is expensive
- output resolution and output-space size are tightly coupled

여기서 로봇은 예시로만 짧게 등장한다.

- robot action prediction is a concrete case where output resolution directly affects control quality

## Paragraph 2: Why larger action resolution may matter

여기서는 쉬운 benchmark가 착시를 줄 수 있다는 점을 말한다.

- coarse discretization can be sufficient on easy tasks
- but this does not imply that moderate resolution is always enough
- precision-sensitive control may require much finer action discretization

이 문단은 Figure 1의 empirical question으로 연결된다.

> Do larger action spaces actually matter for control success?

## Paragraph 3: Why existing heads are problematic in that regime

여기서 기존 접근의 비용 문제를 제시한다.

- dense per-dimension softmax heads require `M` logits per action dimension
- sequential token decoding accumulates latency across tokens
- both become problematic as output resolution grows

핵심 문장:

> The challenge is therefore not only to represent large action spaces, but to do so under realistic compute, latency, and memory budgets.

## Paragraph 4: GVLA overview

여기서 방법을 짧게 소개한다.

- GVLA predicts a structured binary code instead of an `M`-way class per action dimension
- this reduces output dimensionality from `M` to `ceil(log2 M)` per dimension
- the resulting head has more favorable scaling in large action spaces

## Paragraph 5: Code geometry matters

이 문단이 논문을 강하게 만든다.

- simply replacing classes with bits is not enough
- natural binary can map neighboring action bins to distant binary codes
- under bit-wise BCE supervision, that mismatch distorts the optimization signal
- Gray coding preserves local adjacency in code space

이 문단 끝에서 contribution을 명시한다.

## Introduction Contributions

권장 bullet:

1. We identify precision-sensitive control regimes in which substantially larger action resolution is required than standard easy benchmarks suggest.
2. We propose GVLA, a geometry-aware structured binary action head that replaces per-dimension `M`-way prediction with logarithmic-length bit routing.
3. We show that code geometry materially affects high-resolution learning, and that Gray coding substantially improves GVLA in large discrete action spaces.
4. We demonstrate favorable scaling in compute, latency, and head-memory / VRAM footprint relative to dense output heads.

## Method Outline

## 1. Discretized Action Space

각 action 차원을 `M`개 bin으로 양자화하고 action dimension을 `d`라고 두면 전체 이산 output space 크기는

`N = M^d`

이다.

이 정의는 논문 전체의 출발점이다.  
`M`이 커지면 precision은 좋아질 수 있지만, 전체 output space는 매우 빠르게 커진다.

## 2. Dense Action Head

기존 dense head는 각 action 차원마다 `M`개의 logits를 출력하는 per-dimension M-way classifier다.

논문에서는 이걸 다음처럼 부른다.

- `dense per-dimension M-way softmax head`
- 또는 더 짧게 `dense M-way head`

복잡도는 공통 hidden dimension `h`를 포함하면

`O(d h M)`

이고, 본문에서는

`O(dM)`

로 요약해도 충분하다.

## 3. GVLA Head

GVLA는 각 차원의 bin index를 직접 `M`-way classification하지 않고,

`k = ceil(log2 M)`

길이의 binary code로 예측한다.

즉 출력 차원은 차원당 `M`이 아니라 `log2 M`에 비례한다.

핵심 문장:

> GVLA reduces output dimensionality from `M` logits to `log2(M)` bit logits per action dimension.

복잡도는

`O(d h log M)`

또는 hidden dimension을 떼면

`O(d log M)`

이다.

## 4. Quantization Error and High-Resolution Regimes

균일 quantization에서 각 차원 bin width는

`Delta = 2 / M`

이고, scalar quantization error는 대략

`|u - u_hat| <= 1 / M`

수준이다.

이 직관이 precision regime 실험과 연결된다.

- 낮은 `M`에서는 quantization error가 크다
- precision-sensitive regime에서는 이 오차가 실제 성공률 저하로 이어질 수 있다
- 따라서 충분히 큰 `M`이 실제로 필요할 수 있다

## 5. Bit-Wise Objective

GVLA의 학습 objective는 bit-wise BCE다.

`L_bit = sum_{j=1..d} sum_{ell=1..k} BCE(g_{j,ell}(z), c_ell(a_j))`

직교 regularization까지 포함하면

`L = L_bit + lambda * L_ortho`

로 쓸 수 있다.

이 수식의 역할은 “우리가 단순 softmax classifier가 아니라, bit target geometry 자체를 설계하는 structured prediction 문제를 푼다”는 점을 명확히 하는 것이다.

## 6. Natural Binary vs Gray Code

여기서 논문의 geometry contribution이 나온다.

natural binary에서는 neighboring bins가 code-space에서 멀어질 수 있다.

예:

`3 = 011`

`4 = 100`

즉 인접한 bin인데 Hamming distance가 3이다.

이런 carry boundary에서는 작은 action perturbation이 bit-space에서는 큰 jump가 된다.

반면 Gray code는

`g(i) = i XOR (i >> 1)`

로 정의되며,

`H(g(i), g(i+1)) = 1`

을 만족한다.

### Proposition

모든 `i >= 0`에 대해 Gray code는 adjacent bins를 exactly one-bit transition으로 보낸다.

이 proposition은 full theorem이 아니라 locality statement로 충분하다.  
중요한 것은 proof 자체보다, 이 성질이 bit-wise BCE supervision과 직접 연결된다는 점이다.

## 7. Memory / VRAM Implications

이 논문은 시간 복잡도만이 아니라 메모리 문제도 다룬다.

dense head는 head parameter, output activation, gradient, optimizer state가 large output dimension과 함께 커진다.  
반면 GVLA는 `log` 길이 output만 유지하면 되므로 head memory growth가 훨씬 완만하다.

따라서 논문에서는 다음 문장을 직접 써도 된다.

> As output resolution grows, dense heads scale poorly not only in computation but also in output dimensionality and head memory, whereas structured heads require only logarithmic output size.

## Experiments Outline

본문 실험은 세 축만 남긴다.

1. high-resolution action spaces can matter
2. code geometry matters
3. structured heads scale better in the large-output regime

## Figure 1. Why High-Resolution Action Spaces Matter

이 그림은 문제 정의와 동기부여를 담당한다.

### Panel A. Coarse discretization hurts

소스:

- `pick_place_can` 기본 quantization 결과

메시지:

- coarse discretization can substantially degrade success
- easy tasks already show that insufficient resolution can fail

### Panel B. Precision-sensitive regime requires larger M

소스:

- `pick_place_can_precision`
- `custom2.5`

권장 포인트:

- `continuous`
- `256`
- `512`
- `1024`
- `2048`

메시지:

- easy tasks can saturate early
- stricter precision regimes may remain unsolved at `256`
- recovery may appear only at `1024~2048`

이 패널이 없으면 reviewer는 “왜 큰 `M`이 필요한가?”를 바로 묻게 된다.

### Panel C. Large M is expensive

소스:

- `robosuite_study/results.json`
- 필요시 memory companion:
  - `plot_robosuite_budget_comparison.py`
  - `plot_robosuite_results.py`

메시지:

- the regime where success emerges is also the regime where dense heads become expensive
- this cost appears in latency and memory / VRAM feasibility

### Figure 1 Caption Draft

> Fine-grained action resolution can be necessary for precision-sensitive control, but the cost of conventional output heads grows rapidly with action-space size. Left: coarse discretization substantially degrades manipulation success. Center: under a stricter precision-placement regime, performance recovers only at substantially larger action resolution. Right: structured heads remain tractable where dense heads become increasingly expensive in latency and memory.

## Figure 2. Code Geometry Matters

이 그림은 논문의 방법론적 핵심을 보여준다.

권장 구성:

- Dense
- GVLA with natural binary
- GVLA with Gray code

현재 사용 가능한 핵심 포인트:

| M | Dense | GVLA (natural) | GVLA (Gray) |
|---|---:|---:|---:|
| 128 | 10% | 2% | 16% |
| 1024 | 4% | 2% | 18% |

메시지:

- high-resolution GVLA failure는 단순히 binary routing 구조의 문제가 아니다
- target code geometry가 잘못되면 nearby actions가 nearby supervision을 받지 못한다
- Gray coding restores locality and substantially improves high-resolution learning

### Figure 2 Caption Draft

> Preserving locality in code space substantially improves high-resolution structured prediction. Gray coding consistently recovers GVLA performance relative to natural binary targets, indicating that part of the failure mode arises from a mismatch between action-space and code-space geometry.

## Table 1. Precision Regime Validation

이 표는 Figure 1 Panel B를 수치적으로 고정한다.

권장 항목:

- `continuous`
- `256`
- `1024`
- `2048`

필요하면 `512`를 appendix 또는 figure inset에 둔다.

형식:

| Setting | Success Rate | 95% CI |
|---|---:|---:|
| continuous | TBD | TBD |
| GVLA 256 | TBD | TBD |
| GVLA 1024 | TBD | TBD |
| GVLA 2048 | TBD | TBD |

이 표의 목적은 간단하다.

- `10 rollout`에서 보인 방향이 노이즈가 아님을 보여준다
- high-resolution regime의 존재를 메인 claim으로 올릴 수 있게 해준다

## Figure 3. Scaling, Latency, and Memory

이 그림은 시스템 claim을 방어한다.

반드시 아래를 구분해서 써야 한다.

1. head-only latency
2. batch=1 latency
3. larger-batch trend
4. head memory / VRAM budget

즉 이 그림은 “GVLA가 항상 wall-clock에서 더 빠르다”를 주장하는 그림이 아니다.  
이 그림의 목적은 `M` scaling을 제거한다는 점, 그리고 large-output regime에서 dense head의 시간/메모리 비용이 급격히 커진다는 점을 보여주는 것이다.

### Figure 3 Caption Draft

> Structured heads offer favorable scaling as output-space size grows. We report head-only latency, batch-dependent measurements, and head-memory trends separately, since wall-clock ordering can depend on hardware and kernel-launch effects even when asymptotic complexity and memory growth differ substantially.

## Experimental Assets Already Available

현재 바로 사용할 수 있는 자산:

- basic quantization sensitivity
- `custom2.5` precision-regime direction
- Gray code ablation at `M=128, 1024`
- dense vs GVLA latency scaling
- batch-latency sweep
- memory / VRAM budget comparison

즉 breadth는 이미 충분하다.  
남은 일은 새로운 task를 추가하는 것이 아니라, 메인 claim과 직접 연결되는 결과만 높은 신뢰도로 고정하는 것이다.

## Additional Experiments Required Before Submission

## 1. High-confidence precision validation

가장 중요하다.

대상:

- `continuous`
- `256`
- `1024`
- `2048`

권장:

- 최소 `100`
- 가능하면 `200` rollouts
- `95%` binomial confidence interval 포함

## 2. Gray-code 2048 row if feasible

필수는 아니다.  
있으면 더 좋지만, `128`과 `1024`만으로도 메인 Figure 2는 성립한다.

## 3. Synthetic geometry companion

작은 toy figure 하나가 있으면 좋다.

예:

- neighboring bins의 Hamming distance
- natural binary vs Gray code
- nearby actions가 code-space에서 어떻게 보이는지 시각화

이 그림은 논문을 더 NeurIPS스럽게 만들어준다.

## What Goes to the Appendix

메인에는 claim-aligned evidence만 둔다.  
나머지는 가치가 없어서가 아니라 역할이 달라서 appendix로 보낸다.

### Appendix A. AR latency stress

- `0/8/10/15/25/50/75/100ms`
- AR는 좋은 조건에서는 강하지만, decode delay가 커지면 취약하다는 보조 증거

### Appendix B. Token error stress

- `0.02, 0.05, 0.10, 0.20, 0.30`
- 현재 구현에서는 latency가 더 중요한 failure mode였다는 보조 결과

### Appendix C. Full quantization sweeps

- `pick_place_can`
- `pick_place_can_precision` screening

### Appendix D. Full BC sweep

- Dense
- GVLA natural
- GVLA Gray
- full `M` table

### Appendix E. Geometry diagnostics

- orthogonality
- entropy
- collision
- latent partitioning

### Appendix F. Memory / VRAM details

- `plot_robosuite_budget_comparison.py`
- `plot_robosuite_results.py`
- `universal_vla_comparison.py`
- `pi05_verified_gvla_benchmark.py`

메인에서는 memory message를 짧게 쓰고, 구체적 budget lines와 상세 수치는 appendix나 supplement에 두는 편이 좋다.

## Limitations / Discussion

Discussion에서는 솔직해야 한다.

1. GVLA가 continuous policy를 항상 이긴다는 주장 아님
2. GVLA가 batch=1 wall-clock에서 항상 더 빠르다는 주장 아님
3. Gray code가 모든 structured output problem의 최적 coding이라는 주장 아님
4. 현재 contribution은 scalable action head와 code geometry에 있다

즉 논문의 정확한 contribution은:

- large discrete action spaces를 다루는 structured output head
- high-resolution regime의 필요성에 대한 empirical evidence
- geometry-aware coding의 중요성

이다.

## Writing Order

실제 작성은 아래 순서가 좋다.

1. Figure 1 구조 고정
2. Table 1 고표본 결과와 CI 채우기
3. Figure 2 Gray code ablation 고정
4. Figure 3 scaling / memory 그림 정리
5. appendix 표와 보조 그림 정리
6. 마지막으로 초록과 서론 작성

즉 텍스트보다 figure logic을 먼저 고정하는 편이 맞다.

## Final Summary

이 논문은 “GVLA가 로봇을 잘하나?”를 묻는 논문이 아니다.  
이 논문이 묻는 질문은 다음이다.

> When high-resolution action spaces actually matter, how can we model them without paying the full cost of dense prediction?

GVLA의 답은 두 부분으로 이루어진다.

1. structured bit-wise routing으로 large output-space scaling을 줄인다
2. Gray coding으로 code-space locality를 바로잡아 high-resolution learning을 복구한다

이 흐름만 유지되면, 논문은 robotics benchmark paper가 아니라  
**geometry-aware structured output modeling paper with robotics as a demanding application**으로 읽히게 된다.

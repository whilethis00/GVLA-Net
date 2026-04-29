# GVLA: Geometry-Aware Binary Action Routing for Large Discrete Action Spaces

최종 업데이트: 2026-04-30

## Paper Thesis

이 논문은 로봇을 위한 새로운 action head를 소개하는 논문이지만, 더 본질적으로는 큰 structured discrete output space를 어떻게 효율적으로 다룰 것인가에 대한 논문으로 써야 한다.

핵심 명제는 다음과 같다.

정밀 제어에서는 실제로 큰 action resolution이 필요할 수 있다.  
그러나 이런 high-resolution regime에서 기존 dense output head는 계산량, 지연, 메모리 측면에서 빠르게 비싸진다.  
GVLA는 per-dimension `M`-way prediction을 structured binary routing으로 바꾸고, Gray coding으로 code-space locality를 보존함으로써 이 구간을 더 tractable하게 만든다.

이 논문은 precision-cost trade-off를 완전히 없앴다고 주장하지 않는다.  
보다 정확히는, linear output growth를 logarithmic structured prediction으로 대체함으로써 그 trade-off의 Pareto frontier를 더 유리한 방향으로 이동시켰다고 주장한다.

## Title

고정 제목:

> **GVLA: Geometry-Aware Binary Action Routing for Large Discrete Action Spaces**

이 제목은 논문의 세 축을 직접 드러낸다.

- `Geometry-Aware`: code-space locality와 Gray coding
- `Binary Action Routing`: structured bit-wise prediction
- `Large Discrete Action Spaces`: 논문이 직접 다루는 문제

## Abstract Narrative

초록은 다음 흐름을 따라야 한다.

첫 문장은 큰 structured output space를 dense하게 모델링하는 비용 문제를 제시한다.

둘째 문장은 로봇 action prediction을 그 문제의 concrete and demanding application으로 둔다.

셋째 문장에서는 쉬운 benchmark만 보면 moderate action resolution으로 충분해 보일 수 있지만, precision-sensitive regime에서는 더 큰 resolution이 실제로 필요할 수 있음을 제시한다.

넷째 문장에서는 dense head와 sequential decoding이 바로 그 regime에서 비싸진다는 점을 지적한다.

다섯째 문장에서는 GVLA를 소개한다. 각 action dimension에서의 `M`-way prediction 대신 logarithmic-length binary routing을 사용한다는 점을 분명히 적는다.

여섯째 문장에서는 단순 binary coding이 아니라 Gray coding이 locality mismatch를 줄여 high-resolution learning을 개선한다는 점을 적는다.

마지막 문장에서는 실험이 다음 네 축을 뒷받침한다고 정리한다.

- quantization sensitivity
- precision-sensitive recovery at larger `M`
- code-geometry ablation
- scaling in compute, latency, and memory

## Introduction

### 1. Large structured output spaces are expensive

서론은 로봇 이야기로 시작하지 않는다.

먼저 많은 예측 문제가 매우 큰 structured discrete output space를 가진다는 점을 말해야 한다.

출력 해상도를 높일수록 output space는 빠르게 커진다.  
naive dense prediction은 그 공간을 직접 점수화해야 하므로 비싸고, sequential decoding은 개별 출력을 순차적으로 복원하는 과정에서 지연을 누적시킨다.

이 문제는 planning, control, discrete generation 등 다양한 맥락에서 나타나지만, 로봇 action prediction은 그 비용이 실제 제어 품질과 바로 연결된다는 점에서 특히 demanding한 사례다.

### 2. Why larger action resolution may matter

다음 단락에서는 쉬운 benchmark가 주는 착시를 깨야 한다.

coarse discretization은 쉬운 조작 태스크에서는 충분할 수 있다.  
그러나 그것이 곧 모든 manipulation problem에서 moderate action resolution으로 충분하다는 뜻은 아니다.

precision-sensitive regime에서는 충분히 큰 `M`이 실제 성공률과 직접 연결될 수 있다.

이 시점에서 독자에게 던지는 질문은 분명해야 한다.

> 정말 큰 action resolution이 필요할 때, 기존 출력 head는 그 비용을 감당할 수 있는가?

### 3. Why existing output heads become problematic

이후 기존 접근의 비용 문제를 제시한다.

dense per-dimension M-way softmax head는 action dimension마다 `M`개의 logits를 출력해야 한다.  
따라서 `M`이 커질수록 출력 비용이 선형적으로 증가한다.

autoregressive action tokenization은 token들을 순차적으로 예측해야 하므로 총 decoding latency가 누적된다.

즉 문제는 단순히 큰 action space를 표현하는 것이 아니라, 현실적인 compute, latency, memory budget 아래에서 그것을 다루는 것이다.

### 4. GVLA as structured binary action routing

여기서 GVLA를 제시한다.

GVLA는 각 action dimension의 bin을 직접 `M`-way classification으로 고르지 않고, 길이가 `ceil(log2 M)`인 structured binary code로 예측한다.

따라서 각 차원의 출력 차원은 `M`이 아니라 `log2(M)`에 비례한다.

핵심 문장은 다음처럼 짧게 둘 수 있다.

> GVLA reduces output dimensionality from `M` logits to `log2(M)` bit logits per action dimension.

### 5. Why code geometry matters

그러나 단순히 class를 bit로 바꾸는 것만으로는 충분하지 않다.

natural binary에서는 인접한 action bins가 code-space에서는 멀어질 수 있다.  
이 경우 BCE 기반 bit supervision은 물리적으로 가까운 action을 완전히 다른 target처럼 학습하게 된다.

반면 Gray coding은 인접한 bin을 정확히 1-bit transition으로 대응시킨다.  
즉 nearby action perturbation이 nearby code perturbation으로 이어지므로, high-resolution regime에서의 학습 신호가 더 일관적이 된다.

### 6. Quantum-inspired intuition

여기서 양자역학의 직교 측정에 대한 짧은 직관을 넣을 수 있다.

GVLA는 거대한 discrete state를 하나의 거대한 softmax로 판별하는 대신, 여러 개의 orthogonal binary questions로 분해해 판별한다.  
이 관점은 고차원 상태를 여러 개의 직교적 측정 축으로 구분해 가는 양자역학적 intuition과 닮아 있다.

물론 여기서의 주장은 물리학적 동등성이 아니다.  
논문에서 중요한 것은 GVLA가 **orthogonal binary measurements**라는 구조를 통해 큰 output space를 더 효율적으로 분해하고 다룬다는 점이다.

## Contributions

이 논문의 기여는 다음 네 가지로 정리한다.

1. 기존 쉬운 benchmark가 시사하는 것보다 훨씬 더 높은 action resolution이 실제로 필요한 precision-sensitive regime가 존재함을 보인다.

2. 각 action dimension의 `M`-way prediction을 logarithmic-length binary routing으로 대체하는 geometry-aware structured binary action head, GVLA를 제안한다.

3. code geometry가 high-resolution learning의 성패에 실질적 영향을 미치며, Gray coding이 큰 discrete action space에서 GVLA의 성능을 유의미하게 개선함을 보인다.

4. GVLA가 dense output head 대비 compute, latency, 그리고 head-memory / VRAM footprint 측면에서 더 유리한 scaling 특성을 가짐을 보인다.

## Method

### Discretized action space

각 action dimension을 `M`개 bin으로 양자화하고 action dimension을 `d`라고 하면 전체 discrete output space의 크기는

`N = M^d`

가 된다.

`M`을 키우면 quantization error는 줄어들 수 있지만, output space는 매우 빠르게 증가한다.

### Dense action head

기존 dense head는 각 action dimension마다 `M`개의 logits를 출력하는 per-dimension M-way softmax classifier다.

hidden dimension을 `h`라고 할 때, 출력 계산량은 대략

`O(d h M)`

이며, 본문에서는

`O(dM)`

으로 요약해도 충분하다.

### GVLA action head

GVLA는 각 action dimension의 bin index를 직접 `M`-way classification하지 않고, 길이가

`k = ceil(log2 M)`

인 binary code로 예측한다.

이때 출력 계산량은

`O(d h log M)`

또는 단순화하면

`O(d log M)`

이다.

즉 GVLA의 핵심은 “큰 discrete output space를 더 작은 structured code prediction 문제로 바꾼다”는 데 있다.

### Quantization error

균일 quantization에서 각 차원의 bin width는

`Delta = 2 / M`

이고, scalar quantization error는 대략

`|u - u_hat| <= 1 / M`

수준이다.

이 식은 왜 larger `M`이 precision-sensitive regime에서 실제로 중요할 수 있는지를 설명하는 직관적 연결고리다.

### Bit-wise objective

각 action dimension `j`와 bit index `ell`에 대해 예측 logit을 `g_{j,ell}(z)`, target code bit를 `c_ell(a_j)`라고 두면, GVLA의 bit-wise loss는

`L_bit = sum_{j=1..d} sum_{ell=1..k} BCE(g_{j,ell}(z), c_ell(a_j))`

로 쓸 수 있다.

여기에 orthogonality regularization을 더하면

`L = L_bit + lambda * L_ortho`

가 된다.

### Natural binary versus Gray code

natural binary에서는 인접한 action bins가 code-space에서 멀어질 수 있다.

예를 들어

`3 = 011`

`4 = 100`

은 인접한 index이지만 Hamming distance가 3이다.

이런 carry boundary에서는 작은 action perturbation이 bit-space에서는 큰 jump로 바뀐다.

반면 Gray code는

`g(i) = i XOR (i >> 1)`

로 정의되며,

`H(g(i), g(i+1)) = 1`

을 만족한다.

즉 adjacent bins가 code-space에서도 exactly one-bit transition으로 연결된다.

### Memory and VRAM

dense head는 head parameter, output activation, gradient, optimizer state가 output dimension과 함께 커진다.

반면 GVLA는 `log` 길이 output만 유지하면 되므로 head-memory growth가 훨씬 완만하다.

따라서 이 논문은 시간 복잡도뿐 아니라 practical VRAM feasibility까지 함께 다루는 논문이 된다.

### Method takeaway

이 방법 섹션의 결론은 trade-off를 완전히 없앴다는 것이 아니다.

더 정확히는, high-resolution regime에서 precision을 얻기 위해 치러야 하던 dense head의 선형 비용을 logarithmic structured prediction으로 바꾸어, precision-scaling Pareto frontier를 더 유리한 방향으로 이동시켰다는 것이다.

## Experiments

실험 섹션은 세 축만 남긴다.

1. high-resolution action spaces can matter  
2. code geometry matters  
3. structured heads scale better in the large-output regime

### Figure 1. Why high-resolution action spaces matter

Figure 1은 문제 정의와 empirical motivation을 담당한다.

왼쪽 패널에서는 기본 `pick_place_can` quantization 결과를 사용해 coarse discretization이 실제 성공률을 크게 떨어뜨릴 수 있음을 보여준다.

가운데 패널에서는 `pick_place_can_precision`의 `custom2.5` 설정을 사용해, easy task에서는 moderate resolution으로 충분해 보일 수 있지만 stricter precision regime에서는 `256`이 부족하고 `1024~2048`에서야 회복이 시작될 수 있음을 보여준다.

오른쪽 패널에서는 바로 그 regime가 dense head와 sequential decoding에 비용 부담을 만든다는 점을 scaling / memory 관점에서 연결한다.

이 그림의 목적은 “resolution을 키우면 항상 더 좋아진다”를 보여주는 것이 아니다.

더 정확한 메시지는 다음과 같다.

- 낮은 resolution은 실제 실패를 낳을 수 있다
- larger `M`이 필요한 regime가 존재할 수 있다
- 바로 그 regime에서 structured head가 중요해진다

### Figure 2. Code geometry matters

Figure 2는 논문의 방법론적 핵심을 담당한다.

여기서는 Dense, GVLA with natural binary, GVLA with Gray code를 비교한다.

현재 확보된 핵심 포인트는 `M=128`과 `M=1024`이며, 가능하면 `2048`도 추가한다.

이 그림이 보여줘야 하는 것은 단순히 Gray code가 더 좋다는 사실이 아니라, high-resolution GVLA failure의 상당 부분이 target code geometry mismatch에서 왔다는 점이다.

### Table 1. Precision regime validation

Table 1은 Figure 1의 가운데 패널을 수치적으로 고정하는 역할을 한다.

핵심 포인트는

- `continuous`
- `256`
- `1024`
- `2048`

이다.

필요하면 `512`를 appendix 또는 inset으로 둔다.

이 표는 반드시 `100~200` rollouts 수준의 high-confidence validation과 `95%` binomial confidence interval을 포함해야 한다.

### Figure 3. Scaling, latency, and memory

Figure 3는 시스템 claim을 담당한다.

이 그림에서는 반드시 다음을 구분해서 써야 한다.

- head-only latency
- batch=1 latency
- larger-batch trend
- head memory / VRAM budget

이 figure의 목적은 “GVLA가 항상 wall-clock에서 더 빠르다”를 주장하는 것이 아니다.

보다 정확한 목적은,

- dense head는 large-output regime에서 `M`에 선형으로 커지고
- GVLA는 `log M` 구조를 유지하며
- 그 차이가 practical latency와 memory budget에서 드러난다

는 점을 보여주는 것이다.

만약 precision 곡선이 다른 figure에서 충분히 제시되었다면, 이 figure는 success-rate panel 없이 cost-side figure로 유지하는 편이 더 논문답다.

## Main Figures and Tables

메인 본문에는 다음만 남긴다.

- Figure 1: coarse discretization failure + precision-sensitive recovery + cost motivation
- Figure 2: Gray code ablation
- Table 1: high-confidence precision validation with CI
- Figure 3: scaling / latency / memory

## Appendix

appendix에는 메인 claim을 보조하는 전체 자산을 보낸다.

- AR latency stress 전체 표
- token error stress
- 전체 quantization sweep
- full BC sweep
- orthogonality / entropy / collision / geometry diagnostics
- memory / budget comparison details

이것들은 중요하지 않아서 내려가는 것이 아니라, 메인 메시지를 흐리지 않기 위해 역할을 분리하는 것이다.

## Limitations and Discussion

이 논문은 GVLA가 continuous policy를 항상 이긴다고 주장하지 않는다.

GVLA가 batch=1 wall-clock에서 항상 더 빠르다고 주장하지도 않는다.

Gray code가 모든 structured output problem의 최적 coding이라고 말하지도 않는다.

또한 현재 결과만으로 humanoid나 ALOHA로의 일반화를 주장해서는 안 된다.

논문의 기여는 scalable action head와 code geometry의 역할을 large discrete action space 문맥에서 명확히 보여주는 데 있다.

## Writing Order

실제 초안은 figure logic을 먼저 고정하고 쓰는 것이 맞다.

1. Figure 1 구조 확정
2. Table 1의 high-confidence 결과와 CI 채우기
3. Figure 2의 Gray code ablation 고정
4. Figure 3의 scaling / memory figure 정리
5. appendix 표와 보조 그림 배치
6. 마지막으로 초록과 서론 작성

## Closing

이 논문은 “GVLA가 로봇을 잘하나?”를 묻는 논문이 아니다.

이 논문이 묻는 질문은 다음과 같다.

> high-resolution action space가 실제로 필요해지는 순간, 기존 dense prediction의 전체 비용을 치르지 않고 그것을 어떻게 다룰 것인가?

GVLA의 답은 structured bit-wise routing으로 large output-space scaling을 줄이고, Gray coding으로 code-space locality를 복원해 high-resolution learning을 가능하게 하는 것이다.

이 흐름이 유지되면, 논문은 robotics benchmark paper가 아니라 **geometry-aware structured output modeling paper with robotics as a demanding application**으로 읽히게 된다.

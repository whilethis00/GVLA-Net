# NeurIPS Experiment Selection Plan

최종 업데이트: 2026-04-30

## 이 문서의 역할

이 문서는 단순한 실험 체크리스트가 아니다.  
논문을 **NeurIPS 스타일의 large structured output modeling paper**로 쓰기 위해,

- 무엇을 메인 본문에 올릴지
- 무엇을 appendix로 내릴지
- 어떤 실험이 핵심 claim을 직접 지지하는지
- 어떤 실험은 보조 증거이거나 repo 자산으로 남겨야 하는지

를 논문 작성 관점에서 정리한 설계 문서다.

이 문서의 기본 가정은 다음과 같다.

1. 로봇은 핵심 application이지만, 논문의 본체는 robotics method가 아니다.
2. 논문의 중심 문제는 `very large structured discrete output spaces`를 어떻게 효율적으로 모델링할 것인가이다.
3. GVLA의 강점은 단순히 “로봇에서 잘 된다”가 아니라,
   - large output space scaling,
   - structured prediction,
   - coding geometry,
   - compute/latency/memory trade-off
   에 있다.
4. 따라서 실험도 robotics benchmark 나열이 아니라, **geometry + scaling + high-resolution regime**의 세 축으로 재조직되어야 한다.

## 논문의 한 줄 포지셔닝

가장 짧게 쓰면 이 논문은 다음을 주장한다.

> Dense or sequential output prediction becomes increasingly expensive in large discretized action spaces, while high-resolution control can still matter in precision-sensitive regimes. Structured heads and locality-preserving coding provide a more scalable alternative.

한국어로 풀면 다음과 같다.

> 매우 큰 이산 출력 공간에서는 dense 예측이나 긴 순차 디코딩의 비용이 빠르게 증가한다.  
> 반면 정밀한 제어에서는 실제로 높은 action resolution이 필요할 수 있다.  
> 이 구간에서 structured head와 locality-preserving coding이 더 적절한 해법이 된다.

여기서 "비용"은 단순 FLOPs만이 아니라,

- head compute
- latency
- output dimensionality
- head memory / VRAM budget

까지 포함한다.

이 문장을 메인 서론, 초록, 실험 섹션 전체가 지지해야 한다.

## 무엇을 주장하고 무엇을 주장하지 않을 것인가

### 메인으로 주장할 것

1. coarse discretization은 실제 manipulation success를 크게 해칠 수 있다.
2. 일부 precision-sensitive regime에서는 `256` 수준의 action resolution이 충분하지 않을 수 있다.
3. large discrete action/output space로 갈수록 dense head 또는 sequential decoding 비용은 빠르게 증가한다.
4. structured head는 이 large-resolution regime에서 더 유리한 scaling을 가진다.
5. 이 scaling advantage는 compute뿐 아니라 head memory / VRAM footprint에도 해당한다.
6. high-resolution GVLA의 성능 병목 일부는 구조 자체가 아니라 coding mismatch였으며, Gray coding이 이를 완화한다.

### 메인에서 주장하지 않을 것

1. GVLA가 continuous control policy를 전반적으로 이긴다.
2. GVLA가 AR tokenization보다 항상 더 정확하다.
3. 모든 로봇 태스크에서 더 큰 `M`이 반드시 필요하다.
4. `4096+`가 일반적으로 필요하다는 보편 명제.
5. 휴머노이드/ALOHA 결과까지 이미 일반화되었다는 주장.

이 다섯 가지는 지금 데이터로는 방어가 약하다.  
논문은 강한 주장 몇 개를 좁게, 정확하게 밀어야 한다.

## 현재 우리가 가진 가장 강한 실험 자산

### 1. Basic quantization sensitivity

`pick_place_can` 계열 결과는 coarse discretization이 실제 성능을 망칠 수 있다는 아주 직접적인 증거다.

핵심 메시지:

- 낮은 `bins/dim`에서는 성공률이 급감
- 일정 해상도 이상에서야 continuous baseline 회복

이건 메인 Figure의 왼쪽 패널 또는 intro figure 일부로 쓰기 좋다.

### 2. Precision regime with much larger required resolution

`pick_place_can_precision`의 `custom2.5` 설정은 지금 가장 중요한 메인 결과 후보다.

현재 확인된 방향:

- `continuous = 0.6`
- `256 = 0.0`
- `512 = 0.3`
- `1024 = 0.7`
- `2048 = 0.6`
- `4096 = 0.5`

아직 `10-rollout` 기반이라 메인 수치로는 불충분하지만, **“`256`으로는 부족하고 `1024~2048`에서야 회복하는 regime”**를 찾았다는 점이 중요하다.

이건 논문에서 “왜 high-resolution action spaces를 진지하게 다뤄야 하는가”를 보여주는 핵심 empirical anchor다.

### 3. Gray code ablation

기존 `bc_study` 결과는 이미 강하다.

현재 확인된 값:

| M | Dense | GVLA (natural) | GVLA (Gray) |
|---|---:|---:|---:|
| 128 | 10% | 2% | 16% |
| 1024 | 4% | 2% | 18% |

이 표가 중요한 이유는 단순 성능 향상이 아니라, failure mode의 원인을 설명해주기 때문이다.

- natural binary는 인접 bin의 code-space locality를 보존하지 못한다
- BCE target geometry가 action-space locality와 불일치한다
- Gray code는 이를 완화한다

즉 이 실험은 단순 ablation이 아니라, 방법론의 수학적 설계를 뒷받침하는 증거다.

### 4. Scaling / latency

현재 레포에는 두 축의 자산이 있다.

- `experiments/results/robosuite_study/results.json`
- `experiments/results/bc_study/latency_batch.json`

이 자산이 주는 메시지는 분명하다.

- dense head latency는 large `N`에서 빠르게 증가한다
- structured head latency는 훨씬 더 잘 유지된다
- batch regime에 따라 실제 latency ordering은 달라질 수 있으므로, 측정 조건을 명확히 써야 한다

이건 accuracy와 독립적인 시스템 claim을 지지한다.

### 5. Memory / VRAM scaling

메모리 관련 자산도 이미 있다.

- `experiments/plot_robosuite_budget_comparison.py`
- `experiments/plot_robosuite_results.py`
- `experiments/universal_vla_comparison.py`
- `experiments/pi05_verified_gvla_benchmark.py`
- `README.md`의 hardware efficiency 정리

이 자산이 주는 메시지는 분명하다.

- dense head는 large output dimension에서 head memory가 빠르게 증가한다
- structured head는 `log` 길이 code만 유지하면 되므로 memory growth가 훨씬 완만하다
- 실제로 `16 GB`, `40 GB` 같은 practical budget line과 비교하는 자산도 이미 있다

즉 이 논문은 단순히 "latency가 좋은 head"가 아니라, **large output-space regime에서 compute와 VRAM budget을 동시에 맞추는 head**라는 framing이 가능하다.

## NeurIPS형 논문 구조

이 논문은 robotics paper처럼 쓰면 안 된다.  
로봇을 잘 아는 사람에게는 자연스럽더라도, NeurIPS 리뷰어가 보기에 claim이 좁아 보일 수 있다.

권장 구조는 다음과 같다.

### 1. Problem framing

서론에서는 문제를 로봇보다 위로 올려야 한다.

- large structured discrete output spaces are expensive to model densely
- naive code assignments can distort target geometry
- robotics provides a demanding application where these issues become operationally important

즉 로봇은 “왜 이 문제가 중요한가”를 보여주는 응용이어야지, 논문의 유일한 의미가 되어서는 안 된다.

### Intro에서 양자역학 직교 측정 이야기를 할지

넣는 편이 좋다. 다만 비유가 아니라 "영감과 구조" 수준으로 조심해서 써야 한다.

안전한 framing은 다음과 같다.

- GVLA는 각 action 차원을 `M`-way softmax로 직접 분류하는 대신,
  서로 다른 binary decision axis들에 대한 structured measurements로 분해한다.
- 이 관점은 "고차원 상태를 여러 개의 직교적인 질문으로 분해한다"는 점에서
  양자역학의 orthogonal measurement intuition과 닮아 있다.
- 하지만 논문에서 주장해야 하는 것은 물리적 의미가 아니라,
  `orthogonal binary measurements`가 large output spaces에서 더 나은 구조와 scaling을 준다는 계산적 관점이다.

즉 intro에서는

> We draw inspiration from orthogonal measurement views of high-dimensional state discrimination, and use this perspective to decompose large discrete action spaces into structured binary decisions.

정도로 쓰는 게 적절하다.

피해야 할 표현:

- "our method is quantum"
- "this is equivalent to quantum measurement"
- "we derive the model from quantum mechanics"

이런 표현은 오해를 부른다.

### 2. Method

방법 섹션에서는 세 가지를 분리해 써야 한다.

1. structured factorization / GVLA head
2. complexity and scaling intuition
3. coding geometry: natural binary vs Gray code
4. memory / VRAM implications of output dimensionality

특히 Gray code는 부록용 사소한 트릭처럼 쓰면 안 된다.  
high-resolution regime에서 왜 coding geometry가 중요한지를 보여주는 핵심 설계 포인트로 올려야 한다.

### 3. Experiments

실험은 아래 세 줄기로 정리한다.

1. high-resolution regimes can matter
2. structured heads scale better in large output spaces
3. locality-aware coding improves high-resolution structured prediction

즉 “task zoo”가 아니라 “claim-aligned evidence”로 정리해야 한다.

## 본문에 넣을 수식과 이론 포인트

이 논문은 거대한 theorem이 필요한 논문은 아니다.  
대신 아래 세 가지를 간결하게 formalize하면 NeurIPS형 논리 구조가 훨씬 선명해진다.

1. large structured output space의 크기
2. dense vs structured head의 복잡도 차이
3. natural binary와 Gray code의 locality 차이
4. dense vs structured head의 memory growth

## 출력 공간 정의

각 action 차원을 `M`개 bin으로 양자화하고 action 차원이 `d`라고 하자.  
그러면 전체 이산 출력 공간의 크기는

`N = M^d`

이다.

이 식은 단순하지만 중요하다.  
`M`이 커지거나 `d`가 커질수록 전체 출력 공간은 매우 빠르게 커진다.

## Dense head의 복잡도

dense head는 각 action 차원마다 `M`개의 후보를 직접 점수화한다.  
latent 차원을 `h`라고 하면 각 차원당 출력 연산은 대략

`O(hM)`

이고, 전체는

`O(d h M)`

이다.

논문 본문에서는 공통 factor인 `h`를 떼고

`O(dM)`

로 써도 충분하다.

핵심은 해상도 `M`을 키우면 dense 출력 비용이 거의 선형적으로 증가한다는 점이다.

## Structured / GVLA head의 복잡도

GVLA류 structured head는 각 action bin을 직접 `M`-way classification으로 예측하지 않고,  
`k = \lceil \log_2 M \rceil`개의 bit code로 예측한다.

각 action 차원 `a_j`에 대해 code를 `c(a_j) in {0,1}^k`

로 두면, 출력 연산은 각 차원당 대략

`O(hk)`

이고 전체는

`O(d h log M)`

이다.

즉 공통 factor를 떼면

`O(d log M)`

로 쓸 수 있다.

이때 이론적 상대 이점은 대략

`M / log(M)`

에 비례한다.

즉 `M`이 커질수록 structured head의 상대적 이점은 커진다.

### 본문용 핵심 문장

> For a `d`-dimensional discretized action space with `M` bins per dimension, dense heads scale as `O(dM)` while structured bit-wise heads scale as `O(d log M)`. Their relative cost gap therefore widens rapidly in the high-resolution regime.

## Memory / VRAM scaling

출력 차원의 증가는 시간 비용뿐 아니라 메모리 비용도 키운다.

dense head의 마지막 projection이 각 차원마다 `M`개 출력을 가진다고 하면, head parameter 수는 대략

`Theta(d h M)`

이고, 학습 시에는 여기에

- forward activations
- backward gradients
- optimizer states

가 추가된다.

즉 실제 VRAM pressure는 단순 parameter count보다 더 크게 체감된다.

반면 structured head는 각 차원마다 `k = ceil(log2 M)`개의 bit만 출력하면 되므로, head parameter 및 output activation 규모는 대략

`Theta(d h log M)`

이다.

### 본문용 핵심 문장

> As output resolution grows, dense heads scale poorly not only in computation but also in output dimensionality and head memory, whereas structured heads require only logarithmic output size.

## Quantization error와 high-resolution regime

연속 action을 `[-1,1]` 구간에서 균일 quantization한다고 하자.  
차원별 bin width는 대략

`Delta = 2 / M`

이고, scalar action의 quantization error는 대략

`|u - u_hat| <= Delta / 2 = 1 / M`

수준으로 제한된다.

이 식은 precision regime의 직관을 제공한다.

- `M`이 작으면 quantization error가 크다
- precision-sensitive regime에서는 그 오차가 실제 성공률 저하로 이어질 수 있다
- 따라서 충분히 큰 `M`이 실제로 필요할 수 있다

이론적으로는 `M`이 커질수록 quantization error는 줄고, 동시에 output-space cost는 커진다.  
논문은 바로 이 trade-off를 다룬다.

## GVLA 학습 objective

각 action 차원 `j`와 bit index `ell`에 대해 예측 logit을 `g_{j,ell}(z)`, target code bit를 `c_ell(a_j)`라고 하자.  
그러면 GVLA의 bit-wise BCE objective는

`L_bit = sum_{j=1..d} sum_{ell=1..k} BCE(g_{j,ell}(z), c_ell(a_j))`

이다.

직교 regularization까지 포함하면

`L = L_bit + lambda * L_ortho`

로 쓸 수 있다.

이 수식은 “우리가 단순 softmax classifier가 아니라, bit target geometry 자체를 설계하는 structured prediction 문제를 푼다”는 점을 명확히 해준다.

## Natural binary가 왜 local geometry를 깨는가

자연 이진수 encoding을 `b(i)`라고 하자.  
우리가 원하는 것은 action index가 조금 바뀌면 target code도 조금만 바뀌는 것이다.

즉 action-space locality와 code-space locality가 맞아야 한다.

하지만 natural binary에서는 일반적으로

`H(b(i), b(i+1))`

가 작게 유지되지 않는다.

예를 들어,

`3 = 011, 4 = 100`

으로, 인접한 bin임에도 Hamming distance가 3이다.

더 일반적으로 `i = 2^m - 1`일 때는 carry 경계에서

`H(b(i), b(i+1)) = m + 1`

이 된다.

즉 인접한 action bin이 code-space에서는 매우 멀어질 수 있다.

## Gray code가 왜 locality를 보존하는가

Gray encoding을

`g(i) = i XOR (i >> 1)`

로 정의하자.

Gray code의 핵심 성질은

`H(g(i), g(i+1)) = 1`

이라는 점이다.

즉 연속된 두 bin은 code-space에서 정확히 1 bit만 다르다.

### Proposition 1. Locality preservation of Gray coding

모든 정수 `i >= 0`에 대해 Gray code `g(i) = i XOR (i >> 1)`는

`H(g(i), g(i+1)) = 1`

을 만족한다.

증명 아이디어:

- \(i\)와 \(i+1\)는 carry가 발생하는 trailing bits에서만 다르다
- Gray transform은 이러한 carry-induced multi-bit flip을 단일 transition으로 바꾼다
- 따라서 연속된 정수는 Gray space에서 항상 한 bit만 다르게 된다

본문에서는 스케치만 넣고, 정식 증명은 appendix로 보내도 충분하다.

## 왜 BCE target geometry와 직접 연결되는가

bit-wise BCE는 각 bit를 독립적인 label처럼 다룬다.  
따라서 code-space의 Hamming geometry는 곧 학습 신호의 geometry가 된다.

natural binary에서는 nearby action perturbation이 code-space에서는 large Hamming jump로 바뀔 수 있다.  
그러면 “조금 옆 bin으로 가는 것”이 BCE 관점에서는 “많은 bit를 동시에 뒤집는 것”이 된다.

반면 Gray code에서는 nearby bin이 1-bit transition이므로,

- nearby action change
- nearby code change
- smoother BCE supervision

으로 연결된다.

### 본문용 핵심 문장

> Under bit-wise BCE supervision, code-space geometry directly shapes optimization. Gray coding aligns nearby action bins with nearby binary targets, whereas natural binary can assign large Hamming jumps to adjacent actions.

## Precision regime 결과와 수식의 연결

precision task 결과를 설명할 때는 다음처럼 쓰는 게 좋다.

> Increasing `M` reduces quantization error but enlarges the effective structured output space. Our precision-placement experiments show that some regimes remain unsolved at `M=256` and recover only at substantially larger resolution, precisely where structured scaling becomes important.

## 메인 theorem까지는 필요 없고 proposition 수준이면 충분하다

현재 논문에서 정말 필요한 이론 포인트는 아래 두 줄이다.

1. Complexity statement

`Dense = O(dM), Structured = O(d log M)`

2. Gray locality statement

`H(g(i), g(i+1)) = 1`

이 두 가지와 empirical validation이 잘 연결되면, 논문은 “로봇 데모 실험 모음”이 아니라  
`large structured output modeling`에 대한 방법론 논문처럼 보이게 된다.

## 메인 본문에 들어갈 실험

## Figure 1. Why large structured action spaces matter

이 논문의 메인 메시지를 거의 혼자 담당하는 그림이다.

권장 구성은 3패널이다.

### Panel A. Coarse discretization can fail

소스:

- 기존 `pick_place_can` quantization 결과

메시지:

- 단순히 action을 대충 잘게 나누는 것만으로는 안 된다
- low-resolution discretization은 실제 성공률을 크게 떨어뜨릴 수 있다

이 패널의 역할은 “high-resolution을 논문에서 진지하게 다룰 이유”를 제공하는 것이다.

### Panel B. A precision-sensitive regime may require much larger resolution

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

- 쉬운 태스크에서는 `256`이 ceiling일 수 있다
- 하지만 더 엄격한 precision regime에서는 `256`이 부족하고 `1024~2048`에서야 회복이 시작될 수 있다

이 패널은 “large-resolution regime가 실제로 존재한다”는 메인 empirical claim이다.

### Panel C. Scaling / latency cost

소스:

- `robosuite_study/results.json`
- 필요시 `bc_study/latency_batch.json`
- memory / budget companion:
  - `plot_robosuite_budget_comparison.py`
  - `plot_robosuite_results.py`

메시지:

- 그 large-resolution regime를 dense head나 긴 sequential decoding으로 다루면 비용이 빠르게 커진다
- structured heads는 훨씬 유리한 scaling을 가진다
- 이 비용에는 head memory / VRAM feasibility도 포함된다

### 이 Figure의 역할

이 그림 한 장으로 아래 세 문장을 모두 지지해야 한다.

1. 더 높은 resolution이 실제로 필요할 수 있다
2. 큰 action/output space는 계산적으로 비싸다
3. 큰 action/output space는 메모리 측면에서도 빠르게 비싸진다
4. 따라서 structured head가 필요하다

### Figure 1 캡션 초안

> Fine-grained action resolution can be necessary for precision-sensitive control, but the cost of conventional output heads grows rapidly with action-space size. Left: coarse discretization substantially degrades manipulation success. Center: under a stricter precision-placement regime, performance recovers only at substantially larger action resolution. Right: structured heads maintain favorable scaling where dense heads become increasingly expensive in both latency and memory.

## Figure 2. Coding geometry matters

이 그림은 NeurIPS형 포지셔닝에 매우 중요하다.

로봇 결과만 있으면 논문이 “정교한 engineering trick”처럼 보일 위험이 있다.  
Gray code 결과가 들어가면 논문이 `target geometry`를 다루는 structured prediction paper처럼 올라간다.

권장 구성:

- 막대그래프 또는 표: `Dense`, `GVLA natural`, `GVLA Gray`
- `M=128`
- `M=1024`
- `M=2048`은 있으면 추가, 없어도 메인 구성 가능

현재 사용 가능한 메인 포인트:

| M | Dense | GVLA (natural) | GVLA (Gray) |
|---|---:|---:|---:|
| 128 | 10% | 2% | 16% |
| 1024 | 4% | 2% | 18% |

메시지:

- high-resolution GVLA failure의 일부는 head 구조보다 target coding mismatch에 있었다
- locality-preserving coding은 large `M`에서 실제 성능을 회복시킨다

### Figure 2 캡션 초안

> Preserving locality in code space substantially improves high-resolution structured prediction. Gray coding consistently recovers GVLA performance relative to natural binary targets, indicating that part of the high-resolution failure mode arises from a mismatch between action-space and code-space geometry.

## Table 1. Precision regime validation

이 표는 Figure 1 Panel B를 수치적으로 고정하는 역할이다.

대상:

- `custom2.5`
- `continuous`
- `256`
- `1024`
- `2048`
- 필요하면 `512`를 appendix 또는 inset에 유지

현재 메인용 placeholder:

| Setting | Success Rate | 95% CI | Status |
|---|---:|---:|---|
| continuous | TBD | TBD | pending high-sample run |
| GVLA 256 | TBD | TBD | pending high-sample run |
| GVLA 1024 | TBD | TBD | pending high-sample run |
| GVLA 2048 | TBD | TBD | pending high-sample run |

주의:

- `10 rollouts` 결과는 방향 확인용
- 메인 표는 최소 `100`, 가능하면 `200 rollouts`
- low-success regime이므로 반드시 `95%` binomial CI 포함

### 이 Table의 역할

리뷰어가 가장 먼저 물을 질문은 “이 curve가 노이즈가 아닌가?”다.  
이 표는 그 질문을 막는 역할을 한다.

## Figure 3. Scaling and measurement conditions

이 그림은 방법의 시스템적 가치와 complexity claim을 방어한다.

반드시 아래를 명확히 구분해야 한다.

1. head-only latency
2. batch=1
3. larger-batch regime
4. head memory / VRAM budget

즉 이 그림은 “GVLA가 항상 wall-clock에서 더 빠르다”를 말하는 그림이 아니다.  
무엇을 측정했고 무엇을 주장하는지를 정확히 적어야 한다.

권장 구성:

- panel A: head-only latency vs `N`
- panel B: batch size에 따른 latency trend
- panel C 또는 appendix companion: head memory vs `M` or `N`, with practical GPU budget lines

가능한 메시지:

- asymptotic scaling은 GVLA가 훨씬 유리하다
- 실제 hardware regime에서는 launch overhead 등으로 ordering이 달라질 수 있다
- 그러나 large structured output spaces를 dense하게 처리하는 비용이 급격히 커진다는 점은 변하지 않는다
- 그리고 이 비용은 latency뿐 아니라 VRAM feasibility에서도 나타난다

### Figure 3 캡션 초안

> Structured heads offer favorable scaling as output-space size grows. We report head-only latency, batch-dependent measurements, and head-memory trends separately, since wall-clock ordering can depend on hardware and kernel-launch effects even when asymptotic complexity and memory growth differ substantially.

## Appendix로 내려야 할 것

메인 논문에서 중요한 건 “실험이 많다”가 아니라 “주장이 날카롭다”는 점이다.  
따라서 많은 실험은 가치가 없어서가 아니라 역할이 달라서 appendix로 가야 한다.

## Appendix A. AR latency stress

현재 자산:

- `0/8/10/15/25/50/75/100ms`

역할:

- AR tokenization은 좋은 조건에서는 강하다
- 그러나 decode delay가 커지면 빠르게 취약해질 수 있다

메인 본문에서는 한두 문장으로 충분하다.  
전체 표와 sweep는 appendix가 맞다.

## Appendix B. Token error stress

현재 자산:

- `0.02, 0.05, 0.10, 0.20, 0.30`

역할:

- 현재 구현과 태스크에서는 token error보다 latency가 더 중요한 failure mode였음을 보조적으로 보여준다

이건 메인 메시지를 강하게 만들기보다 분산시킬 가능성이 크므로 appendix가 적절하다.

## Appendix C. Basic quantization sweeps

현재 자산:

- `pick_place_can`
- `pick_place_can_precision` 여러 스크리닝 결과

역할:

- 메인 figure를 뒷받침하는 전체 sweep 제공
- main curve에 들어가지 않은 중간 bins와 screening 결과 보존

## Appendix D. Full BC sweep

현재 자산:

- `dense`
- `gvla natural`
- `gvla gray`
- 전 구간 `M` sweep

이건 메인 Figure 2와 Table 1을 보조하는 전체 결과다.

## Appendix E. Geometry / orthogonality / entropy / collision

이 자산은 매우 좋지만, 메인 메시지의 first line claim은 아니다.

역할:

- representation geometry 보조 분석
- orthogonality regularization이 실제로 무엇을 하는지 설명
- bit entropy / collision / latent partitioning 관련 진단

즉 appendix에서는 매우 강하지만, 메인에는 과하다.

## Appendix F. Memory / budget comparison details

메모리 관련 자산은 appendix에서 매우 잘 산다.

- `plot_robosuite_budget_comparison.py`
- `plot_robosuite_results.py`
- `universal_vla_comparison.py`
- `pi05_verified_gvla_benchmark.py`

역할:

- dense와 GVLA의 head memory 계산식 명시
- `16 GB`, `40 GB` budget line 제시
- large-resolution regime에서 dense head가 왜 비현실적인지 보조 설명

메인 본문에는 memory 메시지를 짧게 두고, 구체적인 budget figure는 appendix 또는 supplementary figure로 내려도 충분하다.

## 지금 당장 필요한 추가 실험

새로운 방향의 breadth는 필요 없다.  
필요한 건 지금 있는 핵심 story를 높은 신뢰도로 고정하는 것이다.

## 필수 1. `custom2.5` high-sample validation

가장 중요하다.

목표:

- `continuous`
- `256`
- `1024`
- `2048`

를 최소 `100`, 가능하면 `200 rollouts`로 다시 평가

역할:

- “large-resolution regime exists”를 메인 claim으로 쓸 수 있게 만들어준다

## 필수 2. CI 계산

메인 success-rate 표는 평균만으로 부족하다.

반드시 포함:

- success rate
- `95%` binomial confidence interval

특히 현재처럼 low-success regime에서는 필수다.

## 필수 3. Gray code main ablation 정리

`128`과 `1024` 두 점만으로도 메인 Figure 2는 충분히 구성된다.

`2048`은 있으면 좋지만, 이 실험 하나 때문에 전체 스토리를 흔들 필요는 없다.

## 필수 4. Scaling figure 정리

현재 자산은 충분하다.  
필요한 건 새 측정보다 “무엇을 측정한 그림인지”를 더 명료하게 정리하는 것이다.

여기에는 memory도 포함된다.

- latency는 무엇을 측정했는지
- batch effect는 어떤 의미인지
- memory는 head parameter 기준인지, activation 기준인지, budget overlay인지

를 문서상 분리해서 써야 한다.

## 필수 5. 작은 synthetic geometry 그림

이건 꼭 대규모 새 실험일 필요가 없다.  
오히려 작은 toy visualization이 더 좋다.

예:

- neighboring bins의 Hamming distance
- natural binary vs Gray code
- 1D bin axis 위에서 code-space discontinuity 시각화

이 그림의 역할:

- 로봇 없이도 왜 Gray code가 중요한지 한눈에 설명
- 논문을 robotics paper에서 structured prediction paper로 끌어올림

## 지금은 버릴 것

다음은 지금 논문 메인 제출 관점에서는 욕심이다.

- `4096`를 더 파고드는 일
- precision task를 더 많이 추가하는 일
- ALOHA / humanoid / 14D 확장
- AR 관련 파생 stress를 더 넓히는 일
- continuous baseline과의 정면 승부를 메인 claim으로 올리는 일

이것들은 future work, supplementary, 다음 논문, 혹은 camera-ready 확장으로 남겨두는 편이 낫다.

## 실제 논문 작성 순서

이 문서는 실험 selection 문서지만, 실제로는 writing order까지 결정해야 한다.

권장 순서는 다음과 같다.

1. Figure 1의 3패널 구조를 먼저 확정한다
2. Table 1의 고표본 결과와 CI를 채운다
3. Figure 2 Gray code ablation을 메인으로 고정한다
4. Figure 3 scaling 그림의 측정 조건을 명료하게 다시 쓴다
5. appendix 표와 전체 sweep를 정리한다
6. 그 다음 초록과 서론을 쓴다

즉 텍스트보다 figure logic을 먼저 굳히는 편이 맞다.

## 메인 Figure / Table 배치 요약

### Main

- Figure 1: coarse discretization failure + precision regime recovery + scaling motivation
- Figure 2: Gray code ablation
- Table 1: `custom2.5` high-confidence success rate with CI
- Figure 3: scaling / latency / memory with measurement conditions

### Appendix

- AR latency stress full table
- token error injection
- basic quantization sweeps
- BC full sweep
- geometry / orthogonality / entropy / collision diagnostics
- memory / VRAM budget comparison details
- implementation details

## Reviewer 질문 대응 포인트

이 문서가 필요한 가장 큰 이유는, 실험을 “많이 했다”는 사실보다
리뷰어가 어디를 공격할지 먼저 알고 막기 위함이다.

### 질문 1. “왜 그냥 continuous policy를 쓰지 않나?”

답변 방향:

- 이 논문은 continuous policy를 전반적으로 대체하겠다는 게 아니다
- large structured discrete output spaces를 직접 다룰 때의 scaling 문제를 다룬다
- 로봇 precision regime는 그 문제가 실제로 중요함을 보여주는 응용이다

### 질문 2. “왜 Gray code가 중요한가?”

답변 방향:

- natural binary는 인접한 bin의 locality를 보존하지 않는다
- BCE target geometry가 nearby action perturbation과 불일치한다
- Gray code는 이 mismatch를 줄인다

### 질문 3. “왜 로봇 실험이 필요한가?”

답변 방향:

- large output-space modeling의 비용 문제는 여러 도메인에 존재하지만
- 로봇 제어는 high-resolution action이 실제로 성능과 latency trade-off로 이어지는 demanding application이다

### 질문 4. “왜 실험을 더 많이 안 했나?”

답변 방향:

- 우리는 breadth보다 claim-aligned evidence를 선택했다
- 메인 본문은 geometry, scaling, high-resolution regime에 직접 필요한 실험만 남겼다
- 나머지는 appendix와 repo에 보존했다

## 최종 한 줄 결론

이 논문은 “로봇에서의 한 가지 action head 비교”로 쓰면 약하다.  
반대로,

- high-resolution regime의 존재,
- structured scaling의 필요성,
- coding geometry의 중요성

을 한 줄로 묶으면 NeurIPS형 논문이 된다.

따라서 실험 selection의 원칙은 단순하다.

> **더 많은 태스크를 추가하지 말고,**
> **이미 확보한 핵심 세 축을 더 높은 신뢰도로 고정한 뒤,**
> **로봇을 대표 응용으로 두는 structured output modeling paper로 재구성한다.**

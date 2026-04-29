# GVLA: Geometry-Aware Binary Action Routing for Large Discrete Action Spaces

최종 업데이트: 2026-04-30

## 논문 개요

이 문서는 실험 체크리스트가 아니라, 실제 논문을 써 내려가기 위한 설계 초안이다.  
목표는 GVLA를 단순한 로봇 제어 기법이 아니라, 큰 structured discrete output space를 효율적으로 다루기 위한 방법론으로 정리하는 것이다. 로봇은 이 문제를 가장 분명하게 드러내는 응용이며, 특히 정밀 제어에서는 action resolution, 추론 지연, 메모리 예산이 동시에 얽히기 때문에 이 문제 설정이 자연스럽게 중요해진다.

논문의 중심 질문은 다음과 같다.

> 정밀한 제어를 위해 실제로 큰 action resolution이 필요할 수 있다면, 그 거대한 discrete action space를 dense prediction의 전체 비용을 치르지 않고 어떻게 다룰 것인가?

GVLA의 대답은 두 부분으로 이루어진다. 첫째, 각 action dimension에서 `M`개의 후보를 직접 점수화하는 대신, 길이가 `log2(M)`인 structured binary code를 예측함으로써 출력 차원을 줄인다. 둘째, 단순한 natural binary coding이 아니라 Gray coding을 사용하여 code-space locality를 보존하고, 이를 통해 high-resolution regime에서의 학습 신호를 더 부드럽고 일관되게 만든다.

이 framing에서 중요한 점은, GVLA가 precision과 cost 사이의 trade-off를 완전히 제거한다고 주장하지 않는다는 것이다. 기존 dense head에서는 action resolution `M`을 키울수록 precision은 좋아질 수 있지만, 동시에 출력 비용은 `O(dM)`으로 증가한다. GVLA의 기여는 이 trade-off를 부정하는 것이 아니라, per-dimension `M`-way prediction을 `O(d log M)` structured prediction으로 바꿈으로써 정밀도-비용 Pareto frontier를 더 유리한 방향으로 이동시키는 데 있다. Gray coding은 여기에 더해, high-resolution regime에서 binary code 학습이 locality mismatch 때문에 무너지는 문제를 완화함으로써, 그 shifted frontier 위에서 실제 학습 성능이 유지되도록 돕는다.

## 제목

고정 제목은 다음으로 둔다.

> **GVLA: Geometry-Aware Binary Action Routing for Large Discrete Action Spaces**

이 제목은 논문의 세 축을 그대로 반영한다. `GVLA`는 방법 자체를 지칭하고, `Geometry-Aware`는 Gray code와 locality를 포함한 code-space 설계를 강조하며, `Large Discrete Action Spaces`는 논문의 직접적인 문제 설정을 밝힌다.

## 핵심 주장

이 논문은 다음 하나의 주장을 중심으로 전개된다.

> Precision-sensitive control에서는 큰 action resolution이 실제 성능에 필요할 수 있다. 그러나 이러한 high-resolution regime에서 기존 dense output head나 긴 sequential decoding은 계산량, 지연, 메모리 측면에서 급격히 비싸진다. GVLA는 per-dimension `M`-way prediction을 structured bit-wise routing으로 대체하고, Gray coding을 통해 code-space locality를 정렬함으로써 이 regime를 더 잘 다룬다.

즉 논문의 메시지는 “GVLA가 로봇을 잘한다”가 아니다. 보다 정확히는 “high-resolution action space가 실제로 필요해지는 순간, 기존 출력 head는 비싸지고, GVLA는 그 구간을 tractable하게 만든다”가 중심축이다.

## 초록의 흐름

초록은 서술적으로 다음 흐름을 따라가야 한다.

첫 문장에서는 큰 structured output space를 dense하게 다루는 것이 비싸다는 일반 문제를 제시한다. 그 다음 문장에서는 로봇 action prediction을 이 문제의 구체적이고 demanding한 사례로 위치시킨다. 이어서 기존의 쉬운 benchmark만 보면 moderate action resolution으로 충분해 보일 수 있지만, precision-sensitive regime에서는 훨씬 더 높은 resolution이 실제로 필요할 수 있음을 말한다. 그런 regime에서는 dense head와 sequential decoding이 모두 비싸진다는 점을 지적한 뒤, GVLA를 per-dimension `M`-way prediction 대신 logarithmic-length binary routing을 사용하는 structured head로 소개한다. 마지막으로 Gray coding이 high-resolution learning을 개선한다는 점과, 실험이 quantization sensitivity, precision regime recovery, coding ablation, scaling, 그리고 memory / VRAM advantage를 함께 보여준다는 점으로 마무리한다.

## 서론의 구성

서론은 로봇 이야기로 바로 들어가지 않는다. 먼저 많은 예측 문제가 매우 큰 structured discrete output space를 가지며, 출력 해상도를 높일수록 그 공간은 빠르게 팽창한다는 점을 말해야 한다. 기존 dense prediction은 이 공간을 직접 점수화해야 하므로 비싸고, sequential prediction은 개별 출력을 순차적으로 복원하는 과정에서 지연을 축적한다. 로봇 action prediction은 이 문제를 특히 날카롭게 드러내는 사례인데, 그 이유는 출력 해상도의 부족이 단순한 top-1 error가 아니라 실제 제어 실패로 이어질 수 있기 때문이다.

그 다음 단락에서는 쉬운 benchmark가 주는 착시를 깨야 한다. coarse discretization이 쉬운 태스크에서는 충분할 수 있지만, 그것이 곧 모든 manipulation problem에서 moderate resolution으로 충분하다는 뜻은 아니다. 오히려 precision-sensitive regime에서는 충분히 큰 `M`이 실제 성공률과 직접 연결될 수 있다. 여기서 독자는 자연스럽게 다음 질문으로 넘어가게 된다. “그렇다면 정말 큰 action resolution이 필요할 때, 기존 출력 head는 그 비용을 감당할 수 있는가?”

이후 기존 head의 한계를 정리한다. dense per-dimension M-way softmax head는 action dimension마다 `M`개의 logits를 만들어야 하므로 출력 공간이 커질수록 비용이 선형적으로 증가한다. autoregressive tokenization은 각 token을 순차적으로 생성해야 하므로 총 decoding latency가 누적된다. 이로부터 논문의 핵심 문제가 형성된다. 문제는 단순히 큰 action space를 표현하는 것이 아니라, 현실적인 compute, latency, memory budget 아래에서 그것을 다루는 것이다.

GVLA는 여기서 structured binary action routing이라는 형태로 등장한다. GVLA는 각 action dimension의 bin을 직접 `M`-way classification으로 고르지 않고, 길이가 `ceil(log2 M)`인 binary code를 예측한다. 이 표현은 큰 output space를 더 효율적으로 다룰 수 있게 하지만, 단순히 bit 수를 줄이는 것만으로는 충분하지 않다. 바로 이 지점에서 code geometry가 핵심이 된다.

이 시점에서 양자역학의 직교 측정에 대한 짧은 직관을 넣을 수 있다. GVLA는 거대한 discrete state를 하나의 거대한 softmax로 판별하는 대신, 여러 개의 orthogonal binary questions로 분해해 판별한다. 이 관점은 고차원 상태 구분을 직교 측정의 조합으로 바라보는 양자역학적 intuition과 닮아 있다. 물론 여기서의 주장은 물리적 동등성이 아니라 계산적 구조에 관한 것이다. 즉 GVLA는 orthogonal binary measurements라는 관점을 통해 large output space를 더 잘 분해하고 다루는 structured prediction 방식으로 제시된다.

서론의 마지막 단락은 natural binary와 Gray code의 차이를 소개해야 한다. natural binary에서는 인접한 action bin이 code-space에서는 멀어질 수 있다. 이 경우 BCE 기반 bit prediction은 물리적으로 가까운 action을 완전히 다른 target처럼 학습하게 된다. Gray coding은 인접한 bin을 정확히 1-bit transition으로 대응시킴으로써 이런 locality mismatch를 줄인다. 이 단락의 끝에서 contribution을 제시한다.

## 기여 요약

논문의 기여는 네 가지로 정리할 수 있다.

첫째, 기존의 쉬운 benchmark가 시사하는 것보다 훨씬 더 높은 action resolution이 실제로 필요한 precision-sensitive control regime가 존재함을 보인다.  
둘째, 각 action dimension의 `M`-way prediction을 logarithmic-length bit routing으로 대체하는 geometry-aware structured binary action head, GVLA를 제안한다.  
셋째, code geometry가 high-resolution learning의 성패에 실질적인 영향을 미치며, Gray coding이 큰 discrete action space에서 GVLA의 성능을 유의미하게 개선함을 보인다.  
넷째, GVLA가 dense output head 대비 compute, latency, 그리고 head memory / VRAM footprint 측면에서 더 유리한 scaling 특성을 가짐을 실험적으로 보인다.

## 방법 섹션의 구성

방법 섹션은 먼저 discretized action space를 정의하는 데서 시작한다. 각 action dimension을 `M`개 bin으로 양자화하고 action dimension을 `d`라고 두면, 전체 discrete output space의 크기는 `N = M^d`가 된다. 이때 `M`을 키우는 것은 quantization error를 줄이는 대신 output space를 매우 빠르게 증가시킨다.

dense action head는 각 action dimension마다 `M`개의 logits를 출력하는 per-dimension M-way softmax classifier로 설명하면 된다. hidden dimension을 `h`라고 할 때 출력 계산량은 대략 `O(d h M)`이고, 본문에서는 `O(dM)`으로 요약해도 충분하다. 이 표현의 목적은 dense head가 `M`에 대해 선형적으로 커진다는 점을 분명히 하는 데 있다.

GVLA는 각 action dimension의 bin index를 직접 `M`-way classification하지 않고, 길이가 `k = ceil(log2 M)`인 binary code로 예측한다. 따라서 각 차원의 출력은 `M` logits가 아니라 `log2(M)` bit logits이 되고, 계산량은 `O(d h log M)` 또는 단순화하면 `O(d log M)`이 된다. 논문에서 가장 짧게 쓰면 다음 문장이 핵심이다.

> GVLA reduces output dimensionality from `M` logits to `log2(M)` bit logits per action dimension.

이후 quantization error와 precision regime를 연결해야 한다. 균일 quantization에서 각 차원의 bin width는 `Delta = 2 / M`이고, scalar quantization error는 대략 `|u - u_hat| <= 1 / M` 수준이다. 이 식은 왜 larger `M`이 precision-sensitive regime에서 실제로 중요할 수 있는지를 설명하는 직관적 연결고리다.

GVLA의 학습 objective는 bit-wise BCE로 정리한다. 각 action dimension `j`와 bit index `ell`에 대해 예측 logit을 `g_{j,ell}(z)`, target code bit를 `c_ell(a_j)`라고 두면, loss는 `L_bit = sum_{j=1..d} sum_{ell=1..k} BCE(g_{j,ell}(z), c_ell(a_j))`로 쓸 수 있다. 여기에 orthogonality regularization을 더해 `L = L_bit + lambda * L_ortho`로 정리하면 된다. 이 수식의 역할은 GVLA가 단순 classifier가 아니라 bit target geometry 자체를 설계하는 structured prediction 방식이라는 점을 분명히 하는 데 있다.

그 다음이 논문의 핵심인 code geometry다. natural binary에서는 인접한 bin이 code-space에서 멀어질 수 있다. 예를 들어 `3 = 011`, `4 = 100`은 인접한 index이지만 Hamming distance가 3이다. carry boundary에서는 이런 multi-bit flip이 더 커질 수 있다. 반면 Gray code는 `g(i) = i XOR (i >> 1)`로 정의되고, `H(g(i), g(i+1)) = 1`을 만족한다. 즉 adjacent bins는 code-space에서도 정확히 한 bit만 다르게 된다. 본문에서는 이것을 proposition 수준으로 간단히 제시하면 충분하다. 증명의 목적 자체보다, 이 성질이 BCE supervision과 직접 연결된다는 점이 더 중요하다.

마지막으로 메모리 논의를 짧게 포함한다. dense head는 head parameter, output activation, gradient, optimizer state가 output dimension과 함께 커진다. 반면 GVLA는 `log` 길이 output만 유지하면 되므로 head memory growth가 훨씬 완만하다. 따라서 논문은 시간 복잡도뿐 아니라 head memory / VRAM feasibility에서도 structured head의 장점을 주장할 수 있다.

이 방법 섹션의 결론은 trade-off를 완전히 없앴다는 것이 아니다. 더 정확히는, high-resolution regime에서 precision을 얻기 위해 치러야 하던 dense head의 선형 비용을 logarithmic structured prediction으로 바꾸어, precision-scaling Pareto frontier 자체를 더 유리한 위치로 이동시켰다는 것이다.

## 실험 섹션의 구성

실험은 세 축만 남긴다. 첫째는 high-resolution action space가 실제로 필요할 수 있다는 점, 둘째는 code geometry가 high-resolution learning에 실질적 영향을 미친다는 점, 셋째는 GVLA가 large-output regime에서 더 유리한 scaling을 가진다는 점이다.

### Figure 1: Why High-Resolution Action Spaces Matter

Figure 1은 문제 정의와 동기를 담당한다. 왼쪽 패널에서는 기본 `pick_place_can` quantization 결과를 사용해 coarse discretization이 실제로 manipulation success를 크게 떨어뜨릴 수 있음을 보여준다. 가운데 패널에서는 `pick_place_can_precision`의 `custom2.5` 설정을 사용해, easy task에서는 moderate resolution으로 충분해 보일 수 있지만 stricter precision regime에서는 `256`이 부족하고 `1024~2048`에서야 회복이 시작될 수 있음을 보여준다. 오른쪽 패널에서는 바로 그 regime가 dense head와 sequential decoding에 비용 부담을 만든다는 점을 scaling / memory 관점에서 연결한다.

이 그림의 역할은 세 문장을 지지하는 것이다. 첫째, 더 높은 resolution은 실제로 필요할 수 있다. 둘째, 큰 action space는 계산적으로 비싸다. 셋째, 따라서 structured head가 필요하다.

### Figure 2: Code Geometry Matters

Figure 2는 논문의 방법론적 핵심을 담당한다. 여기서는 Dense, GVLA with natural binary, GVLA with Gray code를 비교한다. 현재 확보된 핵심 포인트는 `M=128`과 `M=1024`이며, 가능하면 `2048`도 추가할 수 있다. 이 그림이 보여줘야 하는 것은 단순히 Gray code가 “좀 더 좋다”가 아니라, high-resolution GVLA failure의 상당 부분이 binary routing 구조 자체가 아니라 target code geometry mismatch에서 왔다는 점이다. 따라서 Figure 2의 메시지는 “high-resolution learning에서 code geometry는 본질적이다”가 되어야 한다.

### Table 1: Precision Regime Validation

Table 1은 Figure 1의 가운데 패널을 수치적으로 고정하는 역할을 한다. 핵심 포인트는 `continuous`, `256`, `1024`, `2048`이고, 필요하면 `512`를 appendix나 inset으로 둔다. 이 표는 반드시 `100~200` rollouts 수준의 high-confidence validation과 `95%` binomial confidence interval을 포함해야 한다. 리뷰어가 가장 먼저 공격할 부분은 “이 curve가 단지 10 rollout noise가 아니냐”이므로, 이 표는 그 질문을 막는 기능을 가진다.

### Figure 3: Scaling, Latency, and Memory

Figure 3는 시스템 claim을 담당한다. 여기서는 반드시 무엇을 측정한 그림인지 구분해서 써야 한다. head-only latency, batch=1 latency, larger-batch trend, 그리고 head memory / VRAM budget을 구분해서 제시한다. 이 그림의 목적은 “GVLA가 항상 wall-clock에서 더 빠르다”를 주장하는 것이 아니라, dense head가 large-output regime에서 `M`에 선형으로 커지는 반면 GVLA는 `log M` 구조를 유지한다는 점, 그리고 그 차이가 practical memory budget에서도 그대로 드러난다는 점을 보여주는 것이다.

## 현재 실험 자산과 배치

메인 본문에는 claim과 직접 연결되는 결과만 남긴다. basic quantization sensitivity, `custom2.5` precision regime, Gray code ablation, dense-vs-GVLA scaling, memory / budget comparison이 메인 자산이다. 반면 AR latency stress, token error stress, 전체 quantization sweep, full BC sweep, geometry diagnostics, orthogonality / entropy / collision 분석은 appendix로 내려간다. 이것들은 중요하지 않아서가 아니라, 메인 메시지를 흐리지 않기 위해 역할을 분리하는 것이다.

## 한계와 논의

Discussion에서는 주장을 넓히지 않아야 한다. 이 논문은 GVLA가 continuous policy를 항상 이긴다고 주장하지 않는다. GVLA가 batch=1 wall-clock에서 항상 더 빠르다고 주장하지도 않는다. Gray code가 모든 structured output problem의 최적 coding이라고 말할 수도 없다. 또한 현재 결과만으로 humanoid나 ALOHA까지 일반화된다고 써서는 안 된다. 이 논문의 기여는 scalable action head와 code geometry의 역할을 large discrete action space 문맥에서 명확히 보여주는 데 있다.

## 작성 순서

실제 초안은 그림 구조를 먼저 고정하고 써야 한다. 먼저 Figure 1의 세 패널 논리를 확정하고, 그 다음 Table 1의 high-confidence 결과를 채운다. 이후 Figure 2의 Gray code ablation을 고정하고, Figure 3의 scaling / memory 그림을 정리한다. appendix 표와 보조 그림을 배치한 뒤 마지막에 초록과 서론을 완성하는 순서가 가장 안정적이다.

## 최종 정리

이 논문은 “GVLA가 로봇을 잘하나?”를 묻는 논문이 아니다. 이 논문이 묻는 질문은 “high-resolution action space가 실제로 필요해지는 순간, 기존 dense prediction의 전체 비용을 치르지 않고 그것을 어떻게 다룰 것인가?”이다. GVLA의 답은 structured bit-wise routing으로 large output-space scaling을 줄이고, Gray coding으로 code-space locality를 복원해 high-resolution learning을 가능하게 하는 것이다.

이 흐름이 유지되면, 논문은 robotics benchmark paper가 아니라 **geometry-aware structured output modeling paper with robotics as a demanding application**으로 읽히게 된다. 보다 직접적으로 말하면, GVLA의 기여는 precision-cost trade-off를 완전히 없애는 것이 아니라, linear output growth를 logarithmic structured prediction으로 대체함으로써 그 Pareto frontier를 더 나은 쪽으로 이동시키는 데 있다.

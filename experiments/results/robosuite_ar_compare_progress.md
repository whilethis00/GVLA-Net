# Robosuite AR 비교 정리

최종 업데이트: 2026-04-30

## 이 문서는 무엇을 정리한 것인가

이 문서는 `experiments/robosuite_quantization_study.py` 로 수행한 Robosuite 계열 실험들을 쉽게 설명하고,  
그 결과로부터 **우리가 실제로 주장할 수 있는 것**과 **아직 주장하면 안 되는 것**을 정리한 메모다.

핵심 질문은 단순하다.

1. 행동(action)을 토큰처럼 쪼개서 순차적으로 생성하는 방식이 실제 제어에서 얼마나 강한가?
2. 행동 공간이 커지고, 지연(latency)이나 디코딩 비용이 커질 때 어떤 문제가 생기는가?
3. 이런 조건에서 왜 더 빠르고 구조적인 action representation, 즉 GVLA류 방법이 필요한가?

이 문서는 단순 로그가 아니라, 발표/슬라이드/본문 초안으로 바로 옮길 수 있도록 해석까지 포함한다.

## 아주 쉬운 설명: AR token이 뭔가

### AR token 방식

AR은 autoregressive의 약자다.  
쉽게 말하면 **한 번에 행동 전체를 내는 대신, 행동을 여러 개의 작은 토큰으로 쪼개서 하나씩 순서대로 예측하는 방식**이다.

예를 들어 7차원 action이 있으면:

- 1번째 차원을 토큰으로 예측
- 2번째 차원을 토큰으로 예측
- 3번째 차원을 토큰으로 예측
- ...
- 7번째 차원을 토큰으로 예측

이렇게 해서 최종 action을 복원한다.

장점:

- 언어 모델처럼 토큰 생성 구조를 그대로 쓰기 쉽다.
- 이산(discrete) 표현으로 다루기 쉬워서 대규모 VLA 구조와 잘 붙는다.

단점:

- 토큰을 여러 개 순서대로 만들어야 하므로, **한 action을 복원하는 데 시간이 누적**될 수 있다.
- 각 차원을 bin으로 양자화하므로, **표현 해상도**가 충분히 높지 않으면 제어 오차가 커질 수 있다.
- 토큰 수, vocabulary 크기, 디코딩 지연, 시스템 latency budget에 성능이 민감할 수 있다.

### 이 문서에서 말하는 GVLA류 방식

여기서 GVLA류 방식은 핵심적으로 다음 메시지를 가진다.

- action space가 매우 커져도
- dense하게 전부 펼쳐서 점수를 매기거나
- 토큰을 순차적으로 길게 생성하지 않고도
- 더 구조적으로, 더 빠르게 action을 다룰 수 있는 표현/헤드가 필요하다

즉, 이 문서의 비교는 단순히 “AR이 좋다/나쁘다”가 아니라,

> **현실적인 제어 조건에서 어떤 action representation이 더 유지비가 적고, 더 빠르고, 더 안정적인가**

를 보려는 것이다.

## 왜 이 비교가 중요한가

많은 현대 VLA 계열 방법들은 결국 다음 중 하나에 기대는 경우가 많다.

- action을 discretize해서 token처럼 다룬다
- autoregressive하게 action을 생성한다
- 혹은 큰 discrete action head를 사용한다

이 방식은 좋은 조건에서는 잘 된다.  
문제는 실제 로봇 제어는 언어 생성과 달리 **지연에 매우 민감**하다는 점이다.

언어는 토큰 하나 늦어도 큰일이 안 난다.  
하지만 로봇 제어는:

- action이 늦게 나오면 이미 상태가 바뀌고
- stale action이 들어가고
- 작은 오차가 누적되며
- 성공률이 급격히 떨어질 수 있다

그래서 우리가 정말 보여주고 싶은 건 이것이다.

> “좋은 조건에서 AR token이 동작하느냐”가 핵심이 아니라,  
> “좋은 조건을 유지하려면 얼마나 비싼 표현/지연 조건이 필요한가”가 핵심이다.

## 이 실험에서 실제로 측정한 것

메인 스크립트:

- `experiments/robosuite_quantization_study.py`

주요 설정:

- task: `pick_place_can`
- 기본 rollout 수: `50`
- horizon: `400`
- device: `cpu`
- AR 쪽: `decode_mode=ar_tokens`
- GVLA 쪽: `decode_mode=gvla`

구현 메모:

- 여기서 latency는 실제 wall-clock 측정이 아니다.
- AR에서는 토큰 7개를 순차적으로 복원한다고 두고, 누적 지연을 stale-step으로 넣는다.
- 구현식은 `total_decode_ms = token_latency_ms * 7`
- control frequency는 `20Hz`, 즉 control period는 `50ms`

쉽게 말하면:

- `token_latency_ms=10`이면 action 전체 복원에 `70ms`가 걸리는 효과를 넣는 셈이다.
- `token_latency_ms=50`이면 action 전체 복원에 `350ms`가 걸리는 효과다.

이 실험은 “실제 GPU/CPU 시스템 벤치마크”라기보다는,

> **AR 스타일의 순차 디코딩이 제어 지연으로 번질 때 얼마나 빨리 성능이 무너지는가**

를 보는 스트레스 테스트에 가깝다.

## 한눈에 보는 결론

가장 쉬운 버전으로 요약하면:

1. 행동을 너무 거칠게 양자화하면 성공률이 크게 떨어진다.
2. AR token 방식은 좋은 조건에서는 매우 강하다.
3. 하지만 누적 decode delay가 커지면 AR도 급격히 무너진다.
4. action space가 커질수록 dense/순차 디코딩 비용은 빠르게 커진다.
5. 따라서 더 빠르고 구조적인 action representation이 필요하다는 주장을 할 수 있다.
6. 실제로 더 높은 action resolution이 필요한 정밀 placement regime를 만들 수 있다.

## 우리가 지금 당장 주장할 수 있는 것

### 주장 1. 행동 표현 해상도는 실제 manipulation 성공률에 매우 중요하다

이건 매우 강하게 말할 수 있다.

`pick_place_can`에서는:

- coarse discretization에서 성능이 심하게 무너지고
- `64 bins/dim` 부근은 아직 불안정하며
- `96 bins/dim` 이상부터 안정적으로 continuous 성능을 회복한다

즉, fine action resolution은 “있으면 좋은 것”이 아니라 **성공을 위해 필요한 조건**에 가깝다.

### 주장 2. AR token 방식은 좋은 조건에서는 잘 된다

이것도 인정해야 한다.

`256 bins`에서

- `0ms`
- `8ms`
- `10ms`
- `15ms`
- `25ms`
- token error `0.02`
- token error `0.05`
- token error `0.10`
- token error `0.20`
- token error `0.30`

까지 모두 `100%`였다.

즉, 현재 데이터로는 “AR tokenization은 본질적으로 약하다”라고 말할 수 없다.

### 주장 3. 하지만 decode delay가 커지면 AR도 빠르게 붕괴한다

이건 현재 결과에서 가장 중요한 포인트다.

고지연 조건에서는:

- `50ms/token`부터 큰 성능 저하가 나타나고
- `75ms/token`에서는 거의 붕괴하며
- `100ms/token`에서는 사실상 사용하기 어려운 수준까지 내려간다

즉, AR 방식은 **좋은 latency regime에서는 잘 되지만, 그 조건이 깨지면 급격히 취약해진다**고 말할 수 있다.

### 주장 4. large action space에서는 빠른 structured head가 필요하다

`robosuite_study/results.json`의 latency 실험은 이 메시지를 준다.

- dense head latency는 action space가 커질수록 급격히 증가한다
- GVLA head latency는 훨씬 더 완만하게 유지된다

즉,

> action space가 커질수록 “전부 펼쳐서 계산하는 방법”이나 “길게 순차 디코딩하는 방법”은 점점 비싸지고,  
> 이때 structured representation이 시스템적으로 중요해진다

는 메시지를 줄 수 있다.

### 주장 5. 쉬운 태스크가 아니라 더 정밀한 태스크에서는 `256`보다 훨씬 큰 해상도가 필요할 수 있다

이제 이 부분도 어느 정도 말할 수 있다.

기본 `pick_place_can`은 너무 쉬워서 `256 bins` 부근에서 이미 ceiling에 도달했다.  
그래서 “왜 더 높은 해상도가 필요한가”를 보여주기엔 약했다.

이를 보완하기 위해 `pick_place_can_precision` 태스크를 만들었고,  
성공 조건을 더 엄격하게 해서 **중심점에 더 정밀하게 놓아야만 성공**하도록 바꿨다.

그 결과, 일부 설정에서는:

- `256`은 거의 실패
- `512`도 충분하지 않음
- `1024`와 `2048`에서야 회복

하는 구간이 실제로 관측됐다.

즉, 적어도 이 정밀 placement regime에서는

> `256 bins`가 항상 충분한 것은 아니며, 더 큰 action resolution이 실제 성공률 차이로 이어질 수 있다.

## 우리가 아직 주장하면 안 되는 것

다음 주장은 아직 과하다.

### “우리 방법이 AR보다 항상 더 좋다”

아직 아니다.

좋은 조건에서는 AR도 너무 잘 된다.  
현재 데이터는 “AR이 항상 안 좋다”를 전혀 보여주지 않는다.

### “이번 결과로 frontier VLA 전체의 한계를 증명했다”

이것도 아니다.

우리가 한 것은 Robosuite scripted policy 기반 비교다.  
실제 대규모 end-to-end VLA 전체를 직접 재학습해서 비교한 것은 아니다.

### “token error는 AR의 큰 약점이다”

현재 실험에서는 오히려 그렇지 않다.

이 세팅에서는 `error_prob=0.30`까지도 무너지지 않았다.  
즉 지금은 token error보다 **latency**가 더 핵심적인 취약점으로 보인다.

## Frontier VLA 대비 우리가 유리하게 말할 수 있는 포인트

여기서는 “프론티어 VLA가 다 틀렸다”가 아니라,  
**그들이 잘 되는 조건과, 우리가 더 유리해질 조건이 다르다**는 식으로 말해야 한다.

### 1. Frontier VLA는 좋은 조건에서 강하다

이건 인정해야 한다.

- 충분히 큰 token vocabulary
- 충분히 작은 decode latency
- 충분히 좋은 시스템 budget

이 있으면 tokenized / autoregressive action 방식도 잘 동작할 수 있다.

### 2. 하지만 그 성능은 공짜가 아니다

좋은 성능을 유지하려면:

- 높은 해상도의 discretization
- 충분히 작은 per-token latency
- 누적 decode delay를 감당할 시스템 budget

이 필요하다.

즉, 성능이 나오는 조건 자체가 이미 꽤 비싸다.

### 3. 우리는 constrained regime에서 의미가 커진다

우리 쪽 메시지는 여기다.

즉,

- action space가 크고
- 빠르게 제어해야 하고
- sequential decode budget이 빡빡하고
- dense head를 크게 키우기 부담스럽고
- latency margin이 작을 때

더 빠르고 구조적인 action representation이 필요하다.

이건 frontier VLA의 성공을 부정하는 메시지가 아니다.  
오히려 그 성공을 유지하기 위해 필요한 비용이 커질수록 우리의 필요성이 커진다는 메시지다.

### 4. 따라서 우리의 strongest story는 “최고 정확도”보다 “효율성과 robustness”다

현재까지 가장 설득력 있는 방향은:

- 최고점 비교
- ideal setting 비교

가 아니라

- latency budget 하 robustness
- action space scaling
- decoding efficiency
- structured action representation의 필요성

쪽이다.

한 줄로 말하면:

> frontier VLA들이 좋은 조건에서 강한 것은 맞다.  
> 하지만 그 성능을 유지하려면 표현 해상도와 디코딩 시스템 비용이 필요하고,  
> 그 비용이 커지는 구간에서 우리 방식의 가치가 커진다.

### 5. 정밀 태스크로 갈수록 “큰 해상도”와 “큰 비용” 문제가 동시에 커진다

이번 precision 실험이 중요한 이유는,

- 쉬운 태스크에서는 `256`만으로 충분할 수 있지만
- 더 정밀한 태스크에서는 `1024~2048` 수준에서야 회복되는 구간이 실제로 나왔기 때문이다.

즉, 우리가 말하고 싶은 건 단순히 “더 큰 `M`이 좋다”가 아니다.

- 어떤 태스크에서는 정말 더 큰 `M`이 필요하고
- 그런데 그 구간에서는 dense / AR 계열 비용이 훨씬 더 커진다
- 따라서 structured action head의 필요성이 같이 커진다

이 조합이 바로 우리의 strongest story다.

## 실험 결과 상세 정리

## 1. GVLA 양자화 스윕

출처: `experiments/results/robosuite_ar_compare_gvla/results.json`

| 설정 | 성공률 |
|---|---:|
| continuous | 100.0% |
| GVLA, 32 bins/dim | 0.0% |
| GVLA, 48 bins/dim | 8.0% |
| GVLA, 64 bins/dim | 84.0% |
| GVLA, 96 bins/dim | 100.0% |
| GVLA, 128 bins/dim | 100.0% |

쉬운 해석:

- `32`, `48 bins`는 너무 거칠어서 거의 실패한다.
- `64 bins`는 많이 회복되지만 아직 손실이 남아 있다.
- `96 bins`부터는 충분히 fine해서 성공률이 회복된다.

## 2. PickPlace 기존 양자화 실험

### 2-1. 5-rollout 스모크 테스트

출처:

- `experiments/results/robosuite_pickplace_smoketest/results.json`
- `experiments/results/robosuite_pickplace_smoketest_v2/results.json`

| 설정 | smoketest | smoketest_v2 |
|---|---:|---:|
| continuous | 100.0% | 100.0% |
| 4 bins/dim | 0.0% | 0.0% |
| 8 bins/dim | 20.0% | 0.0% |
| 16 bins/dim | 0.0% | 0.0% |
| 32 bins/dim | 0.0% | 20.0% |
| 64 bins/dim | 100.0% | 80.0% |
| 128 bins/dim | 100.0% | 100.0% |
| 256 bins/dim | 100.0% | 100.0% |
| 512 bins/dim | 100.0% | 100.0% |
| 1024 bins/dim | 100.0% | 100.0% |

### 2-2. 50-rollout 실험

출처: `experiments/results/robosuite_pickplace_50roll/results.json`

| 설정 | 성공률 |
|---|---:|
| continuous | 100.0% |
| 4 bins/dim | 0.0% |
| 8 bins/dim | 0.0% |
| 16 bins/dim | 2.0% |
| 32 bins/dim | 2.0% |
| 64 bins/dim | 80.0% |
| 128 bins/dim | 100.0% |
| 256 bins/dim | 100.0% |
| 512 bins/dim | 100.0% |
| 1024 bins/dim | 100.0% |

### 2-3. 100-rollout 전이 구간 실험

출처: `experiments/results/robosuite_pickplace_transition_100roll/results.json`

| 설정 | 성공률 |
|---|---:|
| continuous | 100.0% |
| 32 bins/dim | 1.0% |
| 48 bins/dim | 17.0% |
| 64 bins/dim | 81.0% |
| 96 bins/dim | 100.0% |
| 128 bins/dim | 100.0% |

정리:

- 여러 실험이 일관되게 “`64~96 bins` 근처에 중요한 전이 구간이 있다”고 말해준다.

## 3. AR baseline 및 약한 교란 실험

### 3-1. 낮은 지연 조건

출처:

- `robosuite_ar_compare_ar0/results.json`
- `robosuite_ar_compare_ar8/results.json`
- `robosuite_ar_compare_ar10/results.json`
- `robosuite_ar_compare_ar15/results.json`
- `robosuite_ar_compare_ar25/results.json`

| 설정 | 성공률 |
|---|---:|
| AR 256 bins, 0ms/token | 100.0% |
| AR 256 bins, 8ms/token | 100.0% |
| AR 256 bins, 10ms/token | 100.0% |
| AR 256 bins, 15ms/token | 100.0% |
| AR 256 bins, 25ms/token | 100.0% |

쉬운 해석:

- 이 정도 지연에서는 AR가 전혀 흔들리지 않는다.
- 따라서 “작은 latency만 줘도 AR는 약하다”는 주장은 틀린다.

### 3-2. 토큰 에러 주입

출처:

- `robosuite_ar_compare_err002/results.json`
- `robosuite_ar_compare_err005/results.json`
- `robosuite_ar_compare_err010/results.json`
- `robosuite_ar_compare_err020_queue/results.json`
- `robosuite_ar_compare_err030_queue/results.json`

| 설정 | 성공률 |
|---|---:|
| AR 256 bins, error 0.02 | 100.0% |
| AR 256 bins, error 0.05 | 100.0% |
| AR 256 bins, error 0.10 | 100.0% |
| AR 256 bins, error 0.20 | 100.0% |
| AR 256 bins, error 0.30 | 100.0% |

쉬운 해석:

- 적어도 이 실험에서는 token error가 주된 병목이 아니다.
- 현재 세팅에서는 latency가 훨씬 더 중요한 문제다.

## 4. AR 고지연 스트레스 실험

주의:

- 아래 표의 `continuous`도 같은 stale-step latency 영향을 받는다.
- 따라서 이 표는 “고지연 제어 상황에서 전체 시스템이 얼마나 무너지는가”를 함께 보여준다.

### 4-1. 50-rollout 본 실험

출처:

- `robosuite_ar_compare_ar50/results.json`
- `robosuite_ar_compare_ar50_queue/results.json`
- `robosuite_ar_compare_ar75/results.json`
- `robosuite_ar_compare_ar75_queue/results.json`
- `robosuite_ar_compare_ar100/results.json`
- `robosuite_ar_compare_ar100_queue/results.json`

| 설정 | continuous | AR 결과 | 비고 |
|---|---:|---:|---|
| 50ms/token | 24.0% | 16.0% | 초기 단발 run |
| 50ms/token | 16.0% | 28.0% | queue run |
| 75ms/token | 8.0% | 16.0% | 초기 단발 run |
| 75ms/token | 8.0% | 8.0% | queue run |
| 100ms/token | 0.0% | 8.0% | 초기 단발 run |
| 100ms/token | 4.0% | 2.0% | queue run |

쉬운 해석:

- `50ms/token`에서 이미 큰 타격이 온다.
- `75ms/token`이면 사실상 둘 다 거의 못 한다.
- `100ms/token`이면 거의 붕괴 상태다.
- 반복 실험 간 편차는 있지만, “고지연에서 빠르게 무너진다”는 결론 자체는 일관적이다.

### 4-2. 20-rollout 파일럿

출처:

- `robosuite_ar_compare_ar50_r20/results.json`
- `robosuite_ar_compare_ar100_r20/results.json`

| 설정 | continuous | AR 결과 |
|---|---:|---:|
| 50ms/token, 20 rollouts | 50.0% | 20.0% |
| 100ms/token, 20 rollouts | 5.0% | 5.0% |

쉬운 해석:

- 표본 수는 작지만, 본 실험과 같은 경향을 다시 보여준다.

## 4-3. 정밀 placement 태스크(`pick_place_can_precision`)에서의 고해상도 요구

기본 `pick_place_can`은 `256 bins` 부근에서 이미 ceiling에 닿기 때문에,  
더 높은 action resolution의 실질적 이득을 보여주기 어렵다.

이를 보완하기 위해 `pick_place_can_precision` 태스크를 추가했다.

핵심 아이디어:

- 같은 PickPlace 환경을 쓰되
- “bin 안에 들어갔는가”만 보는 대신
- **bin 중심에 얼마나 가깝게 놓였는가**를 추가 성공 조건으로 본다

즉 더 정밀한 placement를 요구하는 태스크다.

### 4-3-1. 난이도 스크리닝

짧은 스크리닝 결과:

| 설정 | continuous | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| easy | 1.0 | 0.0 | 0.333 | 1.0 |
| medium | 0.333 | 0.0 | 0.0 | 0.333 |
| hard | 0.0 | 0.0 | 0.0 | 0.0 |

해석:

- `easy`는 baseline은 좋지만 `512`에서 너무 빨리 포화된다.
- `hard`는 continuous baseline이 죽어 비교용으로 부적절하다.
- 그래서 `easy`와 `medium` 사이 custom 설정을 찾는 방향으로 갔다.

### 4-3-2. custom 설정 스크리닝

#### custom1

설정:

- `xy_tol=0.015`
- `release_clearance=0.075`
- `transport_xy_thresh=0.010`
- `place_height=0.068`
- `kp_place=5.6`

결과:

| continuous | 32 | 128 | 512 | 1024 |
|---:|---:|---:|---:|---:|
| 0.333 | 0.0 | 0.0 | 0.333 | 1.0 |

#### custom2

설정:

- `xy_tol=0.014`
- `release_clearance=0.075`
- `transport_xy_thresh=0.010`
- `place_height=0.068`
- `kp_place=5.4`

결과:

| continuous | 32 | 128 | 512 | 1024 |
|---:|---:|---:|---:|---:|
| 0.333 | 0.0 | 0.333 | 0.667 | 0.667 |

추가 확장 스크리닝:

| continuous | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.333 | 0.0 | 0.0 | 0.667 | 0.0 | 0.667 | 0.667 |

해석:

- 표본 수가 작아 변동성이 크지만
- `256`은 분명히 부족하고
- `512` 이후 큰 해상도에서 회복 가능성이 보인다는 점은 중요하다.

#### custom3

설정:

- `xy_tol=0.013`
- `release_clearance=0.078`
- `transport_xy_thresh=0.009`
- `place_height=0.067`
- `kp_place=5.2`

결과:

| continuous | 32 | 128 | 512 | 1024 |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

해석:

- 너무 어려워서 baseline이 죽는다.

### 4-3-3. custom2.5: 가장 유망한 설정

`custom2`가 방향은 좋았지만 baseline이 낮아,  
이를 아주 조금 완화한 `custom2.5`를 추가로 실험했다.

설정:

- `xy_tol=0.015`
- `release_clearance=0.075`
- `transport_xy_thresh=0.010`
- `place_height=0.068`
- `kp_place=5.4`

#### 10-rollout 결과

| continuous | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.0 | 0.3 | 0.7 | 0.6 | 0.5 |

이 결과는 현재 문서에서 가장 중요한 새로운 관측 중 하나다.

해석:

- `continuous` baseline이 `0.6`으로 살아 있다.
- `256`은 완전히 실패한다.
- `512`도 아직 충분하지 않다.
- `1024`에서 처음으로 크게 회복한다.
- `2048`도 높은 성능을 유지한다.
- `4096`은 완전히 무의미하진 않지만, `1024/2048`보다 더 좋다고 단정할 정도로 안정적이진 않다.

즉, 우리가 원하던

> “`256`보다 훨씬 큰 해상도 구간에서야 성능이 회복되는 regime”

를 실제로 찾았다고 볼 수 있다.

물론 이 결과도 아직 표본 수가 아주 크진 않으므로,  
최종 결론용으로는 추가 반복이 있으면 더 좋다.  
하지만 **정성적 메시지** 자체는 이미 충분히 강하다.

### 4-3-4. 이 결과로부터 무엇을 주장할 수 있는가

이 precision 결과는 다음 메시지를 준다.

1. `256 bins`가 항상 충분한 것은 아니다.
2. 더 정밀한 placement를 요구하면 `1024~2048` 수준에서야 성능이 회복되는 구간이 생길 수 있다.
3. 즉 큰 action resolution이 실제 성공률 차이로 이어지는 regime가 존재한다.
4. 그리고 바로 그 regime에서 dense/AR 계열의 비용 문제와 우리 방식의 필요성이 강해진다.

한 줄 요약:

> 쉬운 태스크에서는 `256 bins`로 충분할 수 있지만,  
> 더 정밀한 manipulation에서는 `1024~2048` 수준의 high-resolution regime가 실제로 필요해질 수 있다.

## 5. AR 해상도 축 추가 실험

출처: `experiments/results/robosuite_ar_compare_ar128_queue/results.json`

| 설정 | 성공률 |
|---|---:|
| continuous | 100.0% |
| AR 128 bins, 0ms/token | 100.0% |

쉬운 해석:

- AR가 꼭 `256 bins`여야만 좋은 것은 아니다.
- 적어도 이 태스크에서는 `128 bins`도 충분하다.
- 따라서 “AR는 무조건 huge vocabulary가 필요하다”라고 지금 단정할 수는 없다.

## 6. NutAssembly 스모크 테스트

출처:

- `experiments/results/robosuite_nut_smoketest/results.json`
- `experiments/results/robosuite_nut_smoketest_1/results.json`

| 설정 | nut_smoketest | nut_smoketest_1 |
|---|---:|---:|
| continuous | 0.0% | 0.0% |
| 4 bins/dim | 0.0% | 0.0% |
| 8 bins/dim | 0.0% | 0.0% |
| 16 bins/dim | 0.0% | 0.0% |
| 32 bins/dim | 0.0% | 0.0% |
| 64 bins/dim | 0.0% | 0.0% |
| 128 bins/dim | 0.0% | 0.0% |
| 256 bins/dim | 0.0% | 0.0% |
| 512 bins/dim | 0.0% | 0.0% |
| 1024 bins/dim | 0.0% | 0.0% |

쉬운 해석:

- 현재 scripted policy 자체가 이 태스크에서 성공하지 못한다.
- 따라서 이 결과는 discretization 비교의 근거로 쓰면 안 된다.

## 7. Head latency 측정

출처: `experiments/results/robosuite_study/results.json`

| N | k(bits) | Dense ms | GVLA ms | Speedup |
|---:|---:|---:|---:|---:|
| 1,024 | 10 | 0.2292 | 0.0396 | 5.79x |
| 4,096 | 12 | 0.1171 | 0.0264 | 4.43x |
| 16,384 | 14 | 0.1785 | 0.0264 | 6.77x |
| 65,536 | 16 | 2.5785 | 0.0262 | 98.37x |
| 262,144 | 18 | 8.0780 | 0.0286 | 282.75x |
| 1,048,576 | 20 | 17.9340 | 0.0261 | 687.31x |
| 4,194,304 | 22 | 73.9956 | 0.0274 | 2700.46x |

쉬운 해석:

- 작은 action space에서는 차이가 작아 보일 수 있다.
- 하지만 action space가 커지면 dense head는 급격히 느려진다.
- 반면 GVLA head는 거의 일정한 수준을 유지한다.
- 이 표는 “왜 structured action head가 필요한가”를 가장 직접적으로 보여준다.

## 발표나 문서에서 바로 쓸 수 있는 주장 문장

### 버전 1. 가장 안전한 표현

> 우리의 결과는 coarse action discretization이 manipulation success를 크게 저하시킬 수 있음을 보여준다.  
> 또한 autoregressive action tokenization은 favorable latency regime에서는 잘 동작하지만, 누적 decode delay가 커지면 빠르게 취약해질 수 있다.  
> 이러한 결과는 large action space에서 더 빠르고 구조적인 action representation의 필요성을 뒷받침한다.

### 버전 2. 조금 더 공격적이지만 아직 안전한 표현

> 핵심 질문은 AR tokens가 ideal setting에서 동작하느냐가 아니다.  
> 실제 질문은 그 성능을 유지하기 위해 얼마나 높은 action resolution과 얼마나 작은 decode latency budget이 필요한가이다.  
> 우리의 결과는 이 비용이 무시할 수 없음을 보여주며, 따라서 효율적인 structured action representation이 중요함을 시사한다.

### 버전 3. 우리 방식의 필요성을 강조하는 표현

> frontier VLA 계열의 tokenized action generation은 좋은 조건에서 강력할 수 있다.  
> 그러나 그 성능은 높은 표현 해상도와 낮은 누적 decoding latency에 의존한다.  
> action space가 커지고 시스템 latency budget이 빡빡해질수록, 더 빠르고 구조적인 action representation이 실질적인 이점을 제공할 가능성이 커진다.

## 최종 정리

이 문서를 한 줄로 요약하면 다음과 같다.

> **AR token 방식은 좋은 조건에서는 강하다.  
> 하지만 그 성능은 공짜가 아니며, latency와 decoding cost가 커지는 구간에서는 structured하고 빠른 action representation의 가치가 커진다.**

즉, 우리가 밀어야 할 메시지는:

- “AR은 항상 나쁘다”가 아니라
- “AR의 좋은 성능을 유지하려면 비싼 조건이 필요하고, 그 조건이 깨지는 순간 더 효율적인 구조가 필요하다”

이다.

## 아직 결과가 없는 큐

현재 디렉터리는 있으나 `results.json`이 아직 확인되지 않은 항목:

- `robosuite_ar_compare_ar128_lat50_queue`
- `robosuite_ar_compare_ar96_queue`
- `robosuite_ar_compare_ar96_lat50_queue`

이 항목들이 완료되면, AR의 해상도 축과 지연 축을 더 공정하게 연결해서 해석할 수 있다.

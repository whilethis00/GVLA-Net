# Robosuite 후속 실험 계획: "왜 굳이 GVLA인가?"를 설득하기 위한 우선순위

> **핵심 문제:** 지금 결과는 "fine action resolution이 manipulation success에 중요하다"는 점은 보여준다. 그러나 아직 "그래서 왜 반드시 `O(log N)` head가 필요한가?"까지는 완전히 설득하지 못한다.

---

## 현재까지 확보된 결과

`PickPlaceCan` 100-rollout 결과:

| M (bins / dim) | Success Rate |
|:---:|:---:|
| Continuous | 100% |
| 32 | 1% |
| 48 | 17% |
| 64 | 81% |
| 96 | 100% |
| 128 | 100% |

현재 이 결과로 말할 수 있는 것:

- coarse discretization은 실제 manipulation에서 거의 실패한다
- 충분히 fine한 action resolution이 있어야 success가 회복된다
- `PickPlaceCan`에서는 threshold가 대략 `M=64~96` 근처에 있다

하지만 아직 약한 부분:

- "256 bins 같은 기존 discrete head면 이미 충분한 것 아닌가?"
- "fine resolution이 좋다는 건 알겠는데, 왜 굳이 GVLA / `O(log N)`이어야 하는가?"

즉, 다음 실험의 목적은 단순히 `high M is good`를 반복하는 것이 아니라:

> **success가 나오는 resolution regime이 기존 dense head의 현실적 예산 밖에 있다는 점**을 보여주는 것이다.

---

## 전체 전략

후속 실험은 아래 4개 축으로 정리한다.

1. **Budget-Constrained Comparison**
- 가장 우선
- "성공이 나오는 M은 dense head가 감당 가능한가?"를 직접 보여줌

2. **GVLA vs RT-2-Style Tokenization**
- 가장 중요한 후속 축
- "256-bin autoregressive tokenization이면 충분하지 않나?"에 직접 답함

3. **Tolerance-Tightened PickPlace**
- 두 번째 우선
- 더 정밀한 조작일수록 필요한 M threshold가 올라간다는 점을 보여줌

4. **Backbone-Attached Demo**
- 네 번째 우선
- scripted-only 비판을 방어하고 실제 VLA pipeline과 연결

---

## 우선순위 1: Budget-Constrained Comparison

### 질문

> `PickPlaceCan`에서 성공이 발생하는 action resolution은 Dense head의 메모리 / latency 예산 안에 존재하는가?

### 왜 가장 먼저 해야 하나

- 이미 얻은 `PickPlaceCan` 결과를 거의 그대로 활용 가능
- 추가 task 구현이 거의 필요 없음
- "굳이 GVLA?"라는 질문에 가장 직접 답함
- 논문 메인 figure로 바로 연결 가능

### 실험 설계

입력:

- 현재 `PickPlaceCan` success curve
- 기존 Dense vs GVLA head latency benchmark
- Dense / GVLA memory scaling

제약 budget 예시:

- real-time control budget: `10 ms` (`100 Hz`)
- GPU memory budget A: `16 GB`
- GPU memory budget B: `40 GB`

표현 방식:

- x축: `M`
- 왼쪽 y축: success rate
- 오른쪽 y축: Dense head memory 또는 Dense head latency
- feasibility boundary를 수직선 또는 음영 구간으로 표시
- GVLA는 동일 구간에서 feasible로 표시

### 핵심 메시지

> success가 회복되는 `M>=64~96` 구간은 기존 dense head가 현실적인 memory / latency budget 아래에서 다루기 어려운 영역이다.

### 논문용 문장 초안

> Fine action resolution is not merely beneficial; it is required for success. However, the resolution regime where success emerges lies outside the practical feasibility region of a conventional dense action head under realistic control and memory budgets.

### 산출물

- 메인 paper figure 1개
- summary table 1개
- README / docs용 짧은 해석 문단

### 성공 기준

- "success curve"와 "dense feasibility boundary"가 한 그림 안에서 충돌하는 구조가 분명히 보일 것
- 리뷰어가 `why not just use a larger dense head?`를 물었을 때 즉시 답할 수 있을 것

---

## 우선순위 2: Tolerance-Tightened PickPlace

## 우선순위 2: GVLA vs RT-2-Style Tokenization

### 질문

> "256 bins per dimension"이 있는 RT-2-style autoregressive tokenization이면 이미 충분한 것 아닌가? 왜 굳이 GVLA가 필요한가?

### 왜 이 실험이 중요한가

지금까지의 실험은 아래를 잘 보여준다.

- coarse discretization은 실패한다
- fine discretization은 필요하다
- full joint Dense head는 그 high-resolution regime을 감당하기 어렵다

하지만 아직 직접 보여주지 못한 것이 하나 있다.

> **factorized / autoregressive tokenization도 실제 control에서는 충분하지 않을 수 있다**는 점

리뷰어가 가장 날카롭게 공격할 수 있는 지점은 바로 이것이다.

> "좋아, giant Dense softmax는 안 된다는 건 알겠다. 그런데 RT-2나 OpenVLA처럼 각 차원을 256 bins로 따로 tokenization하면 되지 않나?"

따라서 이 실험은 현재 논문의 가장 큰 약점을 정면으로 메우는 역할을 한다.

### 핵심 가설

문제는 단지 `256 bins per dimension`의 표현력이 충분한가가 아니다.
문제는 **그 high-resolution regime을 실제로 low-latency / stable / high-frequency control로 운영할 수 있느냐**이다.

즉:

- RT-2-style tokenization은 resolution 자체는 충분할 수 있다
- 하지만 autoregressive decoding 때문에
  - latency가 길어지고
  - control bandwidth가 제한되고
  - token-level decoding error가 누적되며
  - long-horizon rollout에서 불안정해질 수 있다

반면 GVLA는 같은 high-resolution regime을 한 번의 `O(log N)` routing으로 다룬다.

### 실험 설계

비교 대상:

- **GVLA head**
- **RT-2-style tokenization**
  - 각 action dimension을 `256 bins`로 discretize
  - 이를 token sequence로 autoregressive decoding

가능하면 동일하게 고정할 것:

- same environment
- same observation
- same task instruction
- same action normalization
- same control interface

이상적으로는 같은 latent / backbone 위에서 head만 바꾸는 구조가 가장 좋다.

### 평가 태스크

1. `PickPlaceCan`
- 이미 threshold가 잘 드러나는 task
- 가장 먼저 비교하기 좋음

2. tighter-tolerance `PickPlaceCan`
- placement tolerance를 강화한 버전
- high-resolution / low-latency 차이가 더 크게 날 가능성

### 측정 지표

필수:

- success rate
- final placement error
- mean inference time
- effective control frequency

강하면 좋은 것:

- long-horizon rollout degradation
- error accumulation over steps
- token decoding length vs wall-clock control delay

### 특히 보고 싶은 현상

아래 중 하나만 강하게 보여도 충분히 가치가 있다.

1. **Same resolution, different practicality**
- RT-2-style 256 bins도 표현력은 충분하지만 latency가 커서 control에 불리
- GVLA는 같은 resolution에서 훨씬 빠름

2. **Same budget, different achievable resolution**
- 같은 wall-clock budget 아래 RT-2-style은 낮은 effective action fidelity만 제공
- GVLA는 더 높은 fidelity 유지

3. **Long-horizon instability**
- autoregressive token errors가 rollout 중 누적
- GVLA는 one-shot routing이라 누적 decoding 문제가 없음

### 기대 메시지

> The key issue is not only whether 256 bins per dimension are expressive enough. It is whether a practical action head can operate in that high-resolution regime with sufficient latency, stability, and control bandwidth. RT-2-style tokenization may have adequate nominal resolution, but GVLA is substantially more practical for real-time control.

### 왜 이 실험이 강한가

- current critique를 직접 겨냥한다
- `why GVLA?`에 대해 dense-head argument보다 더 강한 답을 줄 수 있다
- RT-2 / OpenVLA 계열과의 비교 가능성을 열어 준다

### 구현 난이도

중간~높음

이유:

- autoregressive action token decoding 경로가 필요함
- 같은 task에서 head 차이만 비교하도록 맞추는 세팅이 까다로울 수 있음

### 우선 실용적 버전

처음부터 full RT-2 재현을 할 필요는 없다.
가장 실용적인 시작점은 아래다.

- scripted / fixed-latent setting에서
- `GVLA one-shot action decode`
- vs `per-dimension autoregressive token decode`
- latency, control bandwidth, rollout stability 비교

이 간이 비교만으로도 autoregressive control overhead를 꽤 설득력 있게 보여줄 수 있다.

---

## 우선순위 3: Tolerance-Tightened PickPlace

### 질문

> 성공 판정을 더 엄격하게 만들면 필요한 action resolution threshold가 더 올라가는가?

### 왜 필요한가

현재 `PickPlaceCan`은 좋지만, 리뷰어가 여전히 이렇게 생각할 수 있다.

> "그래도 96 bins면 끝난 거 아닌가? 더 어려운 정밀 manipulation에서는 어떨까?"

이 실험은 그 지점을 강화한다.

### 설계 방향

환경은 그대로 유지:

- Robosuite `PickPlace`
- `single_object_mode=2`
- `object_type="can"`
- scripted continuous policy 재사용

바꾸는 것은 success definition:

- 기존: bin 안에 들어가면 성공
- 강화안: target center 근처에 있어야만 성공

예시 tolerance:

- radius `5 cm`
- radius `3 cm`
- radius `2 cm`

또는:

- final xy error threshold
- final pose tolerance

### 기대 결과

- tolerance가 빡세질수록 낮은 `M` 구간 success가 더 빨리 무너짐
- threshold `M*`가 `96`보다 더 올라갈 수 있음

### 핵심 메시지

> precision requirement is task- and tolerance-dependent, and more stringent manipulation criteria push the required discretization resolution further upward.

### 장점

- 새로운 policy를 다시 짤 필요가 거의 없음
- 실험 비용 대비 메시지 강화 효과가 큼

### 산출물

- tolerance별 success curve
- threshold `M*` vs tolerance plot

---

## 우선순위 4: Backbone-Attached Demo

### 질문

> scripted policy가 아니라 실제 VLA backbone에서도 coarse vs fine action resolution 차이가 드러나는가?

### 왜 세 번째인가

- 가장 강한 증거이긴 하지만 구현 비용이 큼
- 지금 논문을 강화하는 데 필수 1순위는 아님
- 앞의 두 실험이 먼저 있어야 backbone demo도 메시지가 선명해짐

### 설계 방향

후보 backbone:

- Octo
- OpenVLA

방법:

- backbone은 고정
- action discretization만 coarse vs fine로 비교
- 가능하면 same environment, same prompt, same rollout protocol

측정 지표:

- success rate
- final distance / placement error
- completion proxy
- mean return

### 기대 메시지

> even with a fixed VLA backbone, finer action discretization improves control fidelity; GVLA matters because it makes that fine-resolution regime computationally practical.

### 주의점

- 이 실험이 scripted threshold 실험을 대체하는 것은 아님
- 오히려 scripted threshold 결과를 backbone 실험으로 bridge하는 용도

---

## 추천 실행 순서

### 필수 순서

1. `Budget-Constrained Comparison`
2. `GVLA vs RT-2-Style Tokenization`
3. `Tolerance-Tightened PickPlace`

### 시간 여유가 있을 때

4. `Backbone-Attached Demo`

### 왜 이 순서인가

- 1번은 가장 싸고 가장 직접적이다
- 2번은 현재 논문의 가장 중요한 약점을 직접 겨냥한다
- 3번은 threshold를 더 올려서 necessity를 강화한다
- 4번은 가장 강하지만 가장 비싸므로 마지막에 붙인다

---

## 각 실험의 기대 기여

| 실험 | 질문 | 기대 효과 | 비용 | 우선순위 |
|------|------|-----------|------|:---:|
| Budget-Constrained Comparison | 성공이 나오는 M이 dense feasible region 안인가? | "왜 GVLA?"에 직접 답함 | 낮음 | **1** |
| GVLA vs RT-2-Style Tokenization | 256-bin autoregressive tokenization이면 충분하지 않나? | 가장 중요한 reviewer doubt 제거 | 중간~높음 | **2** |
| Tolerance-Tightened PickPlace | 더 정밀한 task에선 threshold가 더 커지는가? | high-resolution necessity 강화 | 중간 | **3** |
| Backbone-Attached Demo | 실제 VLA에서도 coarse vs fine 차이가 보이는가? | scripted-only 비판 방어 | 높음 | **4** |

---

## 즉시 실행해야 할 작업

### 1. Budget figure 만들기

- 현재 `PickPlaceCan` 결과 재사용
- Dense memory / latency boundary overlay
- `16 GB`, `40 GB`, `10 ms` 기준선 표시

### 2. Tight tolerance 정의

추천안:

- `r = 5 cm`
- `r = 3 cm`
- `r = 2 cm`

성공 정의를 environment default reward와 별개로 evaluator에서 계산

### 3. RT-2-style comparison 최소 구현안 정리

추천:

- 각 차원당 `256 bins`
- autoregressive decode length를 action dimension 수와 동일하게 설정
- one-shot GVLA와 inference latency / rollout delay 비교

이 단계에서는 "full RT-2 reproduction"보다
"autoregressive tokenized control vs one-shot high-resolution control"의 practical gap을 먼저 보여주는 것이 중요하다.

### 4. Backbone demo 후보 하나 확정

추천:

- 가장 이미 연결이 쉬운 backbone 하나만 선택
- 후보는 `Octo` 우선

---

## 최종 목표

최종적으로 논문이 설득해야 하는 메시지는 아래다.

> Reliable manipulation requires fine action resolution. But the resolution regime where success actually emerges is beyond what conventional dense heads can support under realistic computational budgets. Moreover, even when high nominal resolution is available through autoregressive tokenization, practical control can still be limited by latency and rollout instability. GVLA is valuable not simply because it is faster, but because it is the most practical way to operate in the success-critical high-resolution regime.

---

## 한 문장 결론

> 다음 실험의 목표는 "fine resolution이 좋다"를 반복하는 것이 아니라, **"success가 나오는 resolution regime 자체가 dense head의 현실적 예산 밖에 있다"**는 점을 보여주는 것이다.

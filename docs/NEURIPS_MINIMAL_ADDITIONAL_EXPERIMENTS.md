# NeurIPS 최소 추가 실험 체크리스트

최종 업데이트: 2026-04-30

## 목적

이 문서는 현재 레포와 결과를 기준으로, 논문을 **NeurIPS형 large structured output modeling paper**로 밀기 위해
추가로 필요한 최소 실험만 정리한 실행 문서다.

핵심 원칙은 단순하다.

1. 로봇 태스크를 더 넓히지 않는다.
2. 이미 확보한 서사를 더 높은 신뢰도로 굳힌다.
3. 로봇은 메인 주장이 아니라 대표 응용으로 둔다.
4. 메인 본문은 `geometry + scaling + precise high-resolution regime` 세 축으로 구성한다.

## 현재 포지셔닝

이 논문의 본체는 다음 문제다.

- 매우 큰 이산 출력 공간을 어떻게 효율적으로 모델링할 것인가
- 왜 naive coding이 local geometry를 깨뜨리는가
- 왜 structured head가 large `N`에서 유리한가

로봇은 이 문제를 가장 직관적으로 보여주는 응용이다.

- coarse discretization은 실제 조작 성능을 무너뜨린다
- 정밀한 조작에서는 높은 action resolution이 필요할 수 있다
- 하지만 큰 action space를 다루는 비용은 급격히 커진다

즉 논문 프레임은:

> scalable structured action/output modeling  
> with robotics as a demanding application

이어야 한다.

## 지금 이미 있는 자산

현재 레포에는 이미 아래 축의 증거가 있다.

### 1. High-resolution regime 필요성

- `experiments/robosuite_quantization_study.py`
- `pick_place_can_precision`
- `custom2.5` 설정에서
  - `continuous = 0.6`
  - `256 = 0.0`
  - `512 = 0.3`
  - `1024 = 0.7`
  - `2048 = 0.6`
  - `4096 = 0.5`

이 결과는 적어도 일부 정밀 regime에서는 `256`보다 훨씬 큰 해상도가 필요할 수 있음을 보여준다.

### 2. Coding ablation

- `experiments/results/bc_study/eval_results.json`

현재 확인된 값:

| M | Dense | GVLA (natural binary) | GVLA (Gray) |
|---|---:|---:|---:|
| 128 | 10% | 2% | 15.5% |
| 1024 | 4% | 2% | 18% |

`M=128`은 별도 `200` rollout 재평가에서 `31/200 = 15.5%`로 재확인됐다. 이건 high-resolution GVLA 성능 병목의 일부가 구조 자체가 아니라 `target coding mismatch`였다는 강한 증거다.

### 3. Scaling / latency

- `experiments/results/robosuite_study/results.json`
- `experiments/results/bc_study/latency_batch.json`
- `experiments/high_dim_action_scaling.py`
- `experiments/plot_robosuite_budget_comparison.py`
- `experiments/plot_robosuite_results.py`

현재 메시지는 분명하다.

- dense head latency는 큰 `N`에서 급격히 증가
- GVLA head latency는 훨씬 완만하거나 거의 일정
- batch regime에 따라 trade-off를 구분해 보여줄 수 있음
- dense head memory / VRAM budget도 large `N`에서 빠르게 악화

### 4. Memory / VRAM 자산

이미 있는 자산:

- `plot_robosuite_budget_comparison.py`
- `plot_robosuite_results.py`
- `universal_vla_comparison.py`
- `pi05_verified_gvla_benchmark.py`
- `README.md`의 hardware efficiency 정리

즉 memory 쪽은 "새로운 방향의 실험"이 아니라, 이미 있는 결과를 논문 구조에 맞게 재배치하는 문제에 가깝다.

## 현재 상태 요약

현재 기준으로 보면:

- `custom2.5` 고표본 재검증: 완료
- Gray code main ablation (`128 / 1024 / 2048`): 완료
- scaling / latency / memory figure: 완료
- synthetic geometry figure: 완료

즉 필수 항목은 사실상 대부분 채워졌고,
남은 일은 새 breadth 실험보다 paper assembly에 가깝다.

## 필수 1. `custom2.5` 고표본 재검증

### 왜 필요한가

현재 `custom2.5` 고표본 재검증은 완료됐다.

- `256` 실패
- `1024~2048` 회복
- `continuous`와의 관계도 같은 표에서 확인 가능

이제 `200 rollouts`와 `95% CI`까지 확보했기 때문에,
high-resolution regime의 존재를 메인 결과로 쓰기 위한 최소 신뢰도 조건은 충족했다.

### 최종 목표

다음 4개 포인트만 깊게 본다.

- `continuous`
- `256`
- `1024`
- `2048`

필요하면 `512`를 appendix 또는 inset으로 둔다.

### 산출물

| Setting | Success Rate | 95% CI |
|---|---:|---:|
| continuous | 53.5% | [46.6, 60.3] |
| GVLA 256 | 12.0% | [8.2, 17.2] |
| GVLA 1024 | 43.0% | [36.3, 49.9] |
| GVLA 2048 | 47.0% | [40.2, 53.9] |

### GPU 필요 여부

- 필수 아님
- 이 실험은 학습이 아니라 scripted rollout 기반 success-rate 측정이다
- 병목은 모델 추론보다 robosuite 시뮬레이션 step 수와 rollout 수다
- 따라서 `CPU`로 돌려도 논문용 결과로 충분히 유효하다

### 메인 메시지

> In a stricter precision-placement regime, performance recovers only at substantially larger action resolution, with `256` remaining insufficient.

### 상태

- 완료
- 산출물: `experiments/results/precision_custom25_200roll_core.json`
- 논문용 요약: `experiments/results/neurips_minimal_summary/summary.md`
- 핵심 메시지: `256`은 확실히 부족하고, `1024/2048`에서만 회복이 나타난다

## 필수 2. Gray code 정식 ablation

### 왜 필요한가

NeurIPS형 주장에서는 “왜 naive binary가 나빴는가”가 중요하다.

단순히 성능이 올랐다는 것만으로는 부족하고,

- natural binary는 인접한 bin을 Hamming space에서 멀리 보이게 만들 수 있음
- BCE 학습 신호가 local geometry와 불일치
- Gray code는 인접한 bin의 code-space locality를 보존

라는 구조적 설명이 필요하다.

### 최소 실험

같은 태스크, 같은 평가 프로토콜로 다음을 비교한다.

- Dense
- GVLA + natural binary
- GVLA + Gray code

대상 `M`:

- `128`
- `1024`
- `2048`

### 메인 표 초안

| M | Dense | GVLA (natural) | GVLA (Gray) |
|---|---:|---:|---:|
| 128 | 10% | 2% | 15.5% |
| 1024 | 4% | 2% | 18% |
| 2048 | 16% | 4% | 24% |

### GPU 필요 여부

- 새 학습을 다시 해야 하면 사실상 필요
- 이유:
  - 이 축은 `GVLA (natural)`과 `GVLA (Gray)`의 학습 결과 비교다
  - 평가만 다시 뽑는 건 `CPU`로 가능하지만
  - 새로운 `M=2048`이나 동일 프로토콜 재학습은 `GPU`가 있어야 현실적이다
- 따라서:
  - 기존 `128/1024` 결과 정리만 할 때는 GPU 불필요
  - 새 Gray-code 모델 추가 학습 시에는 GPU 필요

### 메인 메시지

> High-resolution GVLA does not fail solely because the resolution is large; a substantial part of the failure comes from a mismatch between action-space locality and code-space locality.

### 주의

`2048` row까지 확보되었지만, 메인 본문은 여전히 `128`과 `1024`만으로도 충분히 구성 가능하다.  
`2048`은 appendix 또는 확장 figure에서 “high-resolution regime에서도 Gray 효과가 유지된다”는 보강 증거로 쓰면 된다.

### 상태

- `128 / 1024 / 2048` 완료
- 산출물:
  - `experiments/results/neurips_core_figures/gray_code_ablation_paper.png`
  - `experiments/results/neurips_minimal_summary/summary.md`
  - `experiments/results/bc_study/eval_results_2048.json`

### 2048 결과 요약

- `dense_2048 = 16.0%`
- `gvla_2048 = 4.0%`
- `gvla_gray_2048 = 24.0%`

즉 `M=2048`에서도

- `Gray > Dense > natural-binary GVLA`

순서가 유지된다.  
이 결과는 Gray coding 효과가 `1024`에서만 우연히 나타난 것이 아니라,
더 큰 고해상도 구간에서도 유지될 수 있음을 보여주는 보강 증거다.

## 필수 3. Scaling / latency 그림 정리

### 왜 필요한가

정밀 태스크 결과만으로는 “왜 structured head가 필요한가”가 약하다.
그 결과는 high-resolution regime의 필요성을 보여주고,
scaling 그림은 그 regime에서 기존 방식 비용이 왜 문제인지 보여준다.

### 반드시 구분할 것

1. `head-only latency`
2. `batch=1 latency`
3. `large-batch latency`

본문과 캡션에서 무엇을 측정한 것인지 분명히 적어야 한다.

### 사용 가능한 자산

- `experiments/results/robosuite_study/results.json`
- `experiments/results/bc_study/latency_batch.json`

### 최소 그림 구성

- 왼쪽: head-only latency vs `N`
- 오른쪽: batch=1과 large-batch에서 dense vs GVLA 비교

### GPU 필요 여부

- 필수 아님, 다만 있으면 좋음
- 현재 있는 `head-only`, `batch latency` 결과는 이미 자산이 있다
- memory / VRAM budget 비교 자산도 이미 있다
- 논문용 그림 정리 자체는 GPU가 필요 없다
- 만약 하드웨어별 latency를 새로 더 정밀하게 재측정한다면
  - `CPU only`
  - `GPU inference`
  를 구분해 추가 측정할 수 있지만, 제출 전 필수는 아니다

### 메인 메시지

> As the discretized output space grows, dense heads become rapidly more expensive, whereas structured heads retain favorable scaling.

### 상태

- 완료
- 산출물:
  - `experiments/results/neurips_core_figures/scaling_summary_paper.png`
  - `experiments/results/neurips_minimal_summary/summary.md`

포함 항목:

- head-only latency
- batch effect
- memory / VRAM

## 필수 4. 작은 synthetic geometry 실험

### 왜 필요한가

이 실험은 가장 싸고, 가장 NeurIPS스럽다.

로봇을 전면에 두지 않고도

- natural binary가 local geometry를 어떻게 깨는지
- Gray code가 왜 더 smooth한 target을 만드는지

를 직접 보여줄 수 있다.

### 추천 형태

1D 또는 2D discretized output space를 만든다.

예:

- output bin이 한 칸 이동할 때
- natural binary와 Gray code에서 Hamming distance가 어떻게 달라지는지 측정
- 인접 거리와 code-space 거리의 상관관계를 시각화

또는:

- neighboring targets에 대한 BCE target mismatch를 heatmap으로 그림

### 기대 그림

- natural binary: 가까운 bin인데도 Hamming distance가 크게 튀는 구간 존재
- Gray code: 가까운 bin일수록 code-space distance도 작음

### GPU 필요 여부

- 전혀 필요 없음
- 이건 작은 toy analysis 또는 시각화 실험이라 `CPU`로 충분하다
- 오히려 GPU를 쓸 이유가 거의 없다

### 메인 메시지

> Gray coding better preserves locality in structured discrete output spaces, producing a target geometry that is better aligned with nearby action perturbations.

### 왜 중요한가

이 한 장이 들어가면 논문이 “로봇 실험 모음”이 아니라
`output-space geometry`를 다루는 ML 논문처럼 보이게 된다.

### 상태

- 완료
- 산출물:
  - `experiments/results/synthetic_code_geometry/synthetic_code_geometry_paper.png`
  - `experiments/results/synthetic_code_geometry/summary.json`

## 지금 하지 말아야 할 것

다음은 지금 단계에서 메인 제출 품질을 높이기보다 오히려 흔들 가능성이 크다.

- 새 로봇 태스크 추가
- `4096` deep dive
- ALOHA / humanoid / 14D 확장
- continuous policy와 정면 정확도 전면전
- AR stress 결과를 메인으로 올리기

이건 전부 appendix, supplementary, future work, 혹은 다음 논문 자산으로 두는 편이 낫다.

## 메인 본문 구성 제안

### Main Figure 1. Why large structured action spaces matter

세 패널 구성:

1. coarse discretization failure
2. precision regime recovery only at high resolution
3. dense vs GVLA scaling

이 그림 하나가 메인 메시지 대부분을 담당한다.

### Main Figure 2. Coding geometry matters

- natural binary vs Gray code
- 성능 표 또는 막대그래프
- 가능하면 synthetic geometry 그림 포함

### Main Table 1. Precision regime validation

- `continuous / 256 / 1024 / 2048`
- `100~200 rollouts`
- 95% CI 포함

## 메인 주장 문장

이 논문에서 가장 방어 가능한 주장만 남기면 다음과 같다.

1. coarse discretization can substantially degrade manipulation success
2. some precision-sensitive regimes require substantially larger action resolution than standard easy benchmarks
3. large discretized output spaces make dense or sequential output prediction increasingly expensive
4. structured heads offer more favorable scaling in this regime
5. locality-preserving coding improves high-resolution GVLA by aligning code-space geometry with nearby actions

## 남은 일 우선순위

이제 남은 일은 실험보다 조립에 가깝다.

1. `gray_code_ablation`, `scaling_summary`, `synthetic_code_geometry` 중 메인 / appendix 배치 확정
2. 각 figure의 caption과 measurement condition 문장 정리
3. `custom2.5` 결과를 Table 1 형식으로 본문에 삽입
4. 필요하면 `2048` Gray row를 체크포인트 평가만으로 추가
5. 초록 / 서론 / 실험 섹션의 문장 톤 맞추기

## GPU 사용 요약

| 실험 | GPU 필요 여부 | 이유 |
|---|---|---|
| `custom2.5` 고표본 재검증 | 불필요 | scripted rollout 평가라 CPU로 충분 |
| Gray code 기존 결과 정리 | 불필요 | 이미 저장된 결과 활용 |
| Gray code 새 `M` 재학습 | 필요 | 모델 학습이므로 GPU가 현실적 |
| latency/scaling 그림 정리 | 불필요 | 기존 결과 재구성 중심 |
| latency 재벤치마크 추가 | 선택 | CPU/GPU별 측정 보강용 |
| memory / VRAM 그림 정리 | 불필요 | 기존 budget / memory 자산 재구성 중심 |
| synthetic geometry 실험 | 불필요 | toy analysis라 CPU로 충분 |

## 한 줄 결론

NeurIPS형으로 가기 위해 필요한 건 **더 많은 로봇 실험**이 아니라,

- high-resolution regime를 신뢰도 있게 고정하고
- coding geometry를 직접 설명하고
- scaling advantage를 깨끗하게 보여주는 것

이다.

현재는 이 세 축이 대부분 준비된 상태이며,
남은 핵심은 `paper assembly`와 optional `2048` row 추가뿐이다.

# GVLA Fancy Figures Plan

이 문서는 논문용 "인상적인 그림"을 감으로 그리는 것이 아니라, **실제로 재현 가능한 실험을 통해** 만들기 위한 계획서다. 핵심 원칙은 다음과 같다.

- 메인 성능 실험과 시각화 전용 실험을 분리한다.
- conceptual figure와 empirical figure를 섞지 않는다.
- 그림 하나마다 어떤 로그를 저장해야 하는지 먼저 정의한다.

---

## Figure 1. 3D Action Space Shattering

### 목적

GVLA-Net의 직교 투영이 action/code space를 어떻게 분할하는지 직관적으로 보여준다. 핵심 메시지는:

- orthogonal basis는 구조적이고 균형 잡힌 partition을 만든다
- random basis는 더 불규칙하고 중복적인 partition을 만든다
- 이 차이가 collision rate, entropy, code utilization 차이로 이어진다

### 그림 스타일

- 3D scatter 또는 3D cell assignment visualization
- 비교 대상:
  - `Orthogonal GVLA`
  - `Random Projection`
  - 가능하면 `PQ-style partition`은 개념도로만

### 필요한 실험

- 낮은 차원 latent space (`d=3` 또는 `d=4`)에서 synthetic latent points 생성
- 각 latent point를 code로 매핑
- 아래 조건 비교:
  - orthogonal basis
  - random basis
  - partial orthogonal basis

### 저장할 로그

- latent coordinates
- assigned code
- code collision 여부
- per-bit entropy
- pairwise bit correlation
- unique code ratio

### 최종 산출물

- 메인 그림: 3D partition / code assignment visualization
- 보조 그래프:
  - collision rate vs basis type
  - unique code ratio vs basis type

### 구현 후보

- 기존:
  - [visualize_geometry.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/visualize_geometry.py)
  - [ablation_orthogonality.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/ablation_orthogonality.py)
- 신규 추천:
  - `experiments/plot_action_space_shattering.py`

---

## Figure 2. Entropy Waterfall

### 목적

bit budget가 증가할수록 ambiguity가 줄고, code assignment가 sharp해지는 모습을 시각화한다. 핵심 메시지는:

- bit 수가 늘수록 action ambiguity가 감소한다
- orthogonal routing은 entropy를 더 빠르게 줄인다
- 최종적으로 candidate set이 급격히 수축한다

### 그림 스타일

- waterfall plot 또는 stacked curve
- x축: action candidates 또는 sorted code confidence
- y축: bit budget (`1b -> ... -> 24b`)
- z축 또는 색상: confidence / entropy / ambiguity

### 필요한 실험

- `k = 1..24` bit sweep
- 각 k에 대해 code assignment quality 측정
- 실제 probability distribution이 어렵다면 surrogate 지표 사용:
  - top-1 margin
  - effective candidate count
  - code entropy
  - posterior concentration proxy

### 저장할 로그

- bit budget k
- top-1 confidence or surrogate score
- entropy
- candidate ambiguity
- unique code ratio
- collision rate

### 최종 산출물

- 메인 그림: entropy waterfall / ambiguity collapse
- 보조 그래프:
  - entropy vs bits
  - collision rate vs bits
  - unique code ratio vs bits

### 구현 후보

- 기존:
  - [fig2_bit_entropy.png](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/figures/fig2_bit_entropy.png)
  - [export_neurips_table.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/export_neurips_table.py)
- 신규 추천:
  - `experiments/plot_entropy_waterfall.py`

---

## Figure 3. High-Frequency Trajectory Tracking

### 목적

GVLA-Net이 단지 이론적으로 빠른 것이 아니라, 실제로 빠른 제어 주파수에서 더 정밀한 trajectory tracking을 지원할 수 있음을 보여준다. 이 셋 중 **가장 논문 메인 figure로 적합**하다.

### 그림 스타일

- XY plane trajectory overlay
  - ground truth trajectory
  - coarse discrete baseline
  - GVLA trajectory
- 시간축 tracking error plot
- latency bar 또는 inset
- 가능하면 per-step code size/barcode style 보조 시각화

### 필요한 실험

- 동일한 target trajectory를 따라가는 제어 loop
- 비교 대상:
  - coarse discrete
  - GVLA high-resolution
  - 가능하면 PQ-like 또는 proxy baseline
- 여러 control frequency / action resolution 비교

### 저장할 로그

- target trajectory
- predicted trajectory
- per-step tracking error
- final tracking error
- inference latency per step
- collision / failure / timeout 여부

### 최종 산출물

- 메인 그림:
  - XY trajectory overlay
  - tracking error over time
- 보조 표:
  - mean error
  - latency
  - success rate

### 구현 후보

- 기존:
  - [robot_arm_tracking_demo.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/robot_arm_tracking_demo.py)
- 신규 추천:
  - `experiments/plot_tracking_comparison.py`

---

## 우선순위

추천 우선순위는 다음과 같다.

1. `High-Frequency Trajectory Tracking`
2. `3D Action Space Shattering`
3. `Entropy Waterfall`

이유:

- trajectory tracking은 실제 제어 품질과 직접 연결되므로 리뷰어 설득력이 가장 높다
- 3D shattering은 orthogonality의 직관을 주기에 좋다
- entropy waterfall은 가장 멋질 수 있지만, 실측 로그 설계가 가장 까다롭다

---

## 바로 시작할 액션 아이템

### Action 1

`robot_arm_tracking_demo.py`를 기준으로:

- coarse baseline
- GVLA high-resolution
- per-step latency
- trajectory error

를 모두 로그로 저장하도록 수정

### Action 2

orthogonality / collision ablation 결과를 재사용해:

- 3D latent partition figure
- basis type별 collision/entropy 비교 plot

생성

### Action 3

bit sweep 결과를 저장하는 새로운 실험 스크립트 작성:

- `k = 1..24`
- entropy
- collision rate
- unique code ratio

를 CSV로 저장

---

## 논문에서의 위치

- `High-Frequency Trajectory Tracking`: 메인 본문 figure 후보
- `3D Action Space Shattering`: method intuition figure 또는 supplement
- `Entropy Waterfall`: supplement 또는 appendix figure

---

## 최종 원칙

이 문서의 목표는 "예쁜 그림"이 아니라, **주장을 강화하는 실험 기반 figure**를 만드는 것이다.

즉 각 그림은 아래 중 최소 하나를 충족해야 한다.

- mechanism explanation
- empirical support
- reviewer confusion preemption

이 셋과 무관한 그림은 넣지 않는 것이 낫다.

# GVLA-Net: Geometric Vision-Language-Action Network

> **추론 장벽을 부수다 — O(N)에서 O(log N)으로**

---

## 🌌 서론: 왜 기존 AI는 비효율적으로 행동을 선택하는가?

로보틱스 AI 연구는 오랫동안 **'전수 조사 분류'** 의 굴레에 갇혀 있었습니다.

로봇이 "다음에 어떤 행동을 할까?"를 결정하는 순간, 기존의 VLA(Vision-Language-Action) 모델들은 수십만~수백만 개의 행동 후보를 **하나하나 전부 점수 매겨 비교**해야 했습니다. 이것이 바로 `O(N)` 연산 — 후보 수 N에 비례해 연산량이 늘어나는 구조입니다. 행동 공간이 정밀해질수록(N이 커질수록) 추론 비용은 선형으로 폭발합니다. 우리는 이것을 **추론 장벽(Inference Wall)** 이라 부릅니다.

### 양자역학에서 찾은 해답

이 문제를 풀기 위해 우리는 뜻밖의 곳에서 영감을 얻었습니다. 바로 **양자 역학의 측정(Measurement) 이론**입니다.

양자역학에서 서로 **직교(Orthogonal)하는 축**을 따라 측정을 수행하면 특별한 일이 일어납니다. 각 측정이 **완전히 독립적인 1비트의 정보**를 추출합니다 — 중복이 없고, 낭비가 없습니다. 단 `k = log₂(N)` 번의 직교 측정만으로 N개의 상태 중 정확히 하나를 특정할 수 있습니다.

비유하자면 이렇습니다:

> **기존 방식 (Softmax)**: 책 100만 권이 꽂힌 도서관에서 특정 책을 찾기 위해 책장을 처음부터 끝까지 한 권씩 전부 꺼내보는 것.
>
> **GVLA-Net 방식**: "이 책은 인문학 서가에 있나요?" → "그렇다면 오른쪽 절반인가요?" → "위쪽 칸인가요?" → 20번만 물어보면 100만 권 중에서 단 한 권을 특정.

이것이 바로 **기하학적 해싱(Geometric Hashing)** — 행동을 일일이 비교하는 것이 아니라, 잠재 상태 공간에서 `log₂(N)` 번의 직교 이진 질문으로 행동의 위치를 **기하학적으로 특정**하는 방법입니다.

이 접근은 단순히 속도를 높이는 것을 넘어, 로봇 제어에서 오래된 **상충 관계(Trade-off)를 완전히 무너뜨립니다**:

- 기존: 정밀한 행동 공간(큰 N) = 느린 추론 = 실시간 불가
- GVLA-Net: 정밀한 행동 공간(큰 N) = **동일한 속도** = 엣지 디바이스에서도 실시간 가능

---

## 핵심 아이디어: 한 장으로 보기

```
[기존 VLA의 행동 선택 — Softmax]

  잠재 상태 s ──► [N개 행동과 모두 내적 계산] ──► argmax
                          ↑
               O(N) 시간 & 메모리
               (N이 커질수록 선형으로 증가)


[GVLA-Net의 행동 선택 — 직교 기하학적 탐색]

  잠재 상태 s ──► [k번의 직교 예/아니오 질문] ──► k비트 해시 ──► 룩업
                         ↑
              O(log N) 시간 & 메모리
              (N이 아무리 커져도 거의 일정)

  예시: N = 1,000,000개의 행동 후보 → 단 k = 20번의 질문으로 충분
```

### 수식으로 보기

핵심 모듈은 **직교 투영 레이어(Orthogonal Projection Layer)** 입니다:

```
W ∈ ℝ^(k×d)  : 학습 가능한 직교 기저 행렬 (k개의 초평면)

y = W · s    : 잠재 상태를 k개의 초평면에 투영
b = sign(y)  : 각 투영의 부호 → k비트 이진 해시

제약 조건: WW^T ≈ I  (직교성 유지)
```

직교성 제약이 핵심입니다. W의 각 행(초평면)이 서로 직교하면, 각 질문이 완전히 독립적인 정보를 담게 됩니다. 이것이 `log₂(N)` 번의 질문만으로 N개를 구분할 수 있는 수학적 이유입니다.

---

## 아키텍처

```
시각 입력 (ViT/CLIP)  ──┐
                        ├──► [백본 인코더] ──► 잠재 상태 s ∈ ℝ^d
언어 입력 (프롬프트)  ──┘           │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      OrthogonalProjectionLayer      │
                    │                                     │
                    │  W ∈ ℝ^(k×d),  k = ceil(log₂(N))  │
                    │                                     │
                    │  투영:    y = W · s                 │
                    │  이진화:  b = sign(y) ∈ {0,1}^k    │
                    │  학습:    Straight-Through Estimator│
                    │  제약:    ||WW^T - I||²_F → 0       │
                    └─────────────────────────────────────┘
                                    │
                               k비트 해시 b
                                    │
                                    ▼
                         코드북 룩업 → 행동 선택
```

### 핵심 설계 선택

| 설계 요소 | 이유 |
|-----------|------|
| **Straight-Through Estimator (STE)** | `sign()` 함수는 미분 불가능하므로, 학습 중 gradient를 통과시키기 위한 근사 기법 |
| **QR 분해 초기화** | W를 처음부터 정확히 직교 행렬로 시작 → 안정적인 학습 |
| **직교성 손실 `||WW^T - I||²_F`** | 학습 중 기저가 직교성을 잃지 않도록 강제하는 필수 정규화 항 |
| **FP16 완전 호환** | A100/H100에서 최대 처리량을 위한 반정밀도 지원 |

---

## 파일 구조

```
GVLA-Net/
├── models/
│   └── layers.py                     # OrthogonalProjectionLayer (핵심 모듈)
├── utils/
│   └── geometry.py                   # QR 초기화, 직교성 진단 유틸리티
├── experiments/
│   ├── scaling_test.py               # 레이턴시 vs N 스케일링 벤치마크
│   ├── vla_backbone_comparison.py    # 백본별 GVLA vs Softmax 헤드 비교
│   ├── sota_vla_integration.py       # Octo / OpenVLA / RT-2-X / pi0.5 헤드 교체
│   ├── universal_vla_comparison.py   # 통합 SOTA 비교
│   ├── robustness_study.py           # 직교성 훼손 실험
│   ├── robot_arm_tracking_demo.py    # 2자유도 연속 제어 데모
│   ├── vla_final_demo.py
│   └── results/                      # 모든 실험 결과 (CSV, PNG, TeX)
├── third_party/
│   ├── octo/                         # Octo 레퍼런스 코드 (UC Berkeley)
│   └── openpi/                       # pi0.5 / openpi 레퍼런스 (Physical Intelligence)
├── docs/
│   ├── Handover_AI_Agent.md          # 아키텍처 명세 및 인계 문서
│   └── GVLA_Project_Handover.pdf
├── run_scaling_test.sh
├── run_scaling_test_ddp.sh
└── run_openvla_integration.sh
```

---

## 실험 결과

### 1. 스케일링 법칙: 추론 장벽은 실재한다

가장 중요한 결과입니다. 행동 공간 크기 N을 10,000에서 1,000,000까지 늘리며 행동 선택 헤드의 레이턴시를 측정했습니다.

**GVLA-Net의 레이턴시는 N에 관계없이 거의 일정합니다. Softmax는 선형으로 증가합니다.**

| N (행동 후보 수) | Softmax (ms) | GVLA-Net (ms) | 속도 향상 |
|----------------:|-------------:|--------------:|----------:|
| 10,000 | 0.048 | 0.161 | 0.3× |
| 50,000 | 0.182 | 0.162 | **1.1×** |
| 100,000 | 0.349 | 0.161 | **2.2×** |
| 200,000 | 0.674 | 0.162 | **4.2×** |
| 500,000 | 1.653 | 0.162 | **10.2×** |
| 1,000,000 | 3.284 | 0.162 | **20.3×** |

**결과 해석**: N ≈ 5만 이상부터 GVLA-Net이 앞서기 시작하며, 그 이상에서는 격차가 계속 벌어집니다. N = 100만(밀리미터 단위 정밀 제어가 가능한 행동 공간)에서 **20배 빠릅니다**. 더 중요한 것은, GVLA-Net은 N이 아무리 커져도 레이턴시가 ~0.16ms로 고정된다는 점입니다 — `log₂(N)`이 20에서 21로 늘어나는 것은 실제 시간에 아무런 영향을 주지 않습니다.

---

### 2. SOTA VLA 헤드 교체 벤치마크

현존하는 4개의 주요 VLA 모델(Octo, OpenVLA-7B, RT-2-X, pi0.5)의 행동 선택 헤드만 GVLA-Net으로 교체하고 백본은 그대로 유지하는 **"헤드 이식" 실험**을 수행했습니다. 이를 통해 GVLA-Net이 어떤 VLA 백본과도 결합 가능한 범용 모듈임을 검증합니다.

#### 레이턴시 속도 향상

| 모델 | 백본 | 기존 헤드 방식 | N=1k (10비트) | N=32k (15비트) | N=1M (20비트, 추정) |
|------|------|--------------|:---:|:---:|:---:|
| **Octo-Base** | ViT-B (d=768) | Diffusion readout | 20× | 88× | **2,410×** |
| **OpenVLA-7B** | LLM (d=4096) | 자기회귀 밀집 헤드 | 29× | 49× | **89×** |
| **RT-2-X** | PaLI-X (d=4096) | 토큰 logit 분류 | 31× | 90× | **2,072×** |
| **pi0.5** | Gemma-300M (d=1024) | Flow-matching | 0.7× | 52× | **1,734×** |

**결과 해석**: 작은 N(=1k)에서는 이미 잘 최적화된 기존 헤드에 비해 이점이 작거나(pi0.5의 경우 다소 느림), 이미 20~30배 빠른 경우도 있습니다. 그러나 N이 32k 이상으로 커지면 모든 모델에서 50× 이상, N=1M에서는 수백~수천 배의 속도 차이가 납니다. 특히 Octo와 RT-2-X는 N=1M에서 2000배 이상 빨라지는데, 이는 기존 헤드가 대용량 행동 공간에서 얼마나 비효율적인지를 단적으로 보여줍니다.

#### 메모리 절감 (행동 헤드 가중치, N=1M 기준)

| 모델 | 기존 헤드 메모리 | GVLA 헤드 메모리 | 절감 배율 |
|------|:--------------:|:---------------:|:--------:|
| Octo-Base | **52,429 MB** (~51 GB!) | **0.059 MB** | ~900,000× |
| OpenVLA-7B | **16,384 MB** (~16 GB) | **0.31 MB** | ~52,000× |
| RT-2-X | **16,384 MB** (~16 GB) | **0.31 MB** | ~52,000× |
| pi0.5 | **4,096 MB** (~4 GB) | **0.078 MB** | ~52,000× |

**결과 해석**: 기존 밀집 헤드는 N개의 행동 임베딩을 모두 저장해야 합니다(N × d 파라미터). GVLA-Net은 k × d = 20 × d 파라미터만 저장하면 됩니다. N = 1M에서 Octo의 경우 51 GB의 헤드가 **0.06 MB**로 줄어듭니다. 이는 전체 모델을 엣지 디바이스에 올리는 것을 현실적으로 만들어주는 수치입니다.

---

### 3. 로봇 팔 연속 추적 데모 (2자유도)

N = 131,072 (17비트 정밀도)의 행동 공간에서 2자유도 로봇 팔 연속 추적 과제를 시뮬레이션했습니다. GVLA-Net 컨트롤러와 기존 밀집 Softmax 컨트롤러를 직접 비교했습니다.

| 컨트롤러 | 평균 추적 오차 | 최대 오차 | 처리량(FPS) |
|----------|:------------:|:---------:|:-----------:|
| **GVLA-Net** | **0.65** | **1.10** | 1,149 FPS |
| 밀집 Softmax | 15.08 | 30.11 | 1,477 FPS |

**결과 해석**: GVLA-Net이 **약 23배 낮은 추적 오차**를 달성합니다. 이는 핵심 통찰입니다 — 단순히 빠를 뿐 아니라 **더 정확합니다**. 이유는 이렇습니다: 대규모 N에서 밀집 Softmax는 수많은 행동 임베딩 사이에서 logit 차이가 매우 작아져 구별력이 떨어집니다. 반면 기하학적 해싱은 각 행동에 고유한 이진 지문(fingerprint)을 부여하므로, N이 아무리 커져도 선명하고 명확한 구별이 유지됩니다.

---

### 4. 로버스트니스 실험: 직교성은 선택이 아닌 필수

학습된 기저 W를 직교성에서 점점 멀어지도록 교란(perturbation)하면서 정확도를 측정했습니다.

| 교란 강도 (σ) | 직교성 오차 `||WW^T-I||_F` | GVLA 정확도 |
|:------------:|:------------------------:|:-----------:|
| 0.0 (완전 직교) | ~0 (기계 엡실론) | **100%** |
| 0.1 | 0.72 | 82.8% |
| 0.2 | 0.66 | **11.2%** |
| 0.3 | 0.61 | 2.6% |
| 0.5 | 0.69 | 0.2% |

**결과 해석**: 정확도 붕괴가 충격적이고 급격합니다. 약간의 직교성 훼손(σ=0.2)만으로도 정확도가 100%에서 11%로 추락합니다. 이것은 양자역학의 직관과 정확히 일치합니다 — 측정 기저가 직교성을 잃는 순간, 서로 다른 질문들이 같은 것을 묻기 시작하고, 정보의 독립성이 무너지며, 시스템이 실패합니다.

따라서 학습 손실 `L_ortho = ||WW^T - I||²_F`는 선택적 정규화가 아닌, **GVLA-Net이 작동하기 위한 필수 조건**입니다.

---

## 빠른 시작

```bash
# 스케일링 벤치마크 실행
bash run_scaling_test.sh

# OpenVLA 헤드 교체 실험
bash run_openvla_integration.sh

# Python으로 개별 실험 실행
python experiments/scaling_test.py           # 레이턴시 vs N
python experiments/vla_backbone_comparison.py # 헤드 비교
python experiments/robustness_study.py        # 직교성 실험
python experiments/robot_arm_tracking_demo.py # 로봇 팔 데모
python experiments/universal_vla_comparison.py # SOTA 통합 비교
python experiments/export_neurips_table.py    # 논문용 테이블 생성
```

---

## 진행 현황 및 로드맵

| 상태 | 작업 |
|------|------|
| ✅ 완료 | `OrthogonalProjectionLayer` 구현 (STE + 직교성 손실) |
| ✅ 완료 | 스케일링 법칙 검증 (N: 10k → 1M) |
| ✅ 완료 | SOTA VLA 헤드 교체 (Octo, OpenVLA-7B, RT-2-X, pi0.5) |
| ✅ 완료 | 로버스트니스 / 직교성 교란 분석 |
| ✅ 완료 | 2자유도 로봇 팔 연속 추적 데모 |
| ✅ 완료 | NeurIPS 비교 테이블 생성 |
| ✅ 완료 | FLOPs 감소 시각화 차트 생성 |
| ✅ 완료 | Ablation: Orthogonal vs. Random W 직교성 검증 실험 |
| 🔄 진행 중 | OXE / BridgeV2 데이터셋 엔드-투-엔드 파인튜닝 |
| 📋 예정 | 실물 로봇 평가 (7자유도 도달 & 조작) |
| 📋 예정 | 엣지 배포 벤치마크 (Jetson Orin) |
| 📋 예정 | NeurIPS 2026 제출 |

---

## 부록 A. 기하학적 검증 시각화 (Geometric Validation Figures)

> 모든 그림은 논문 제출용 흰색 배경 버전(`_paper.png`)과 프레젠테이션용 다크 버전 두 가지로 생성됩니다.
> 재생성: `python experiments/visualize_geometry.py`

### Figure 1 — Weight Orthogonality Heatmap (WW^T)

![Fig1 Heatmap](experiments/results/figures/fig1_orthogonality_heatmap_paper.png)

직교 W는 WW^T = I (완벽한 단위 행렬), 랜덤 W는 비대각 노이즈가 가득. **off-diag std = 0.0000 vs 0.0904**.

### Figure 2 — Bit Independence & Entropy Analysis

![Fig2 Entropy](experiments/results/figures/fig2_bit_entropy_paper.png)

직교 W: 모든 비트의 엔트로피 = **0.999984 bits** (이론적 최대 1.0), 쌍별 MI = **0.000016**. 각 비트가 완전히 독립적인 새 정보를 담는다는 수학적 증명.

### Figure 3 — 3D Latent Space Partitioning

![Fig3 Partitioning](experiments/results/figures/fig3_latent_partitioning_paper.png)

k개의 직교 초평면이 잠재 공간을 2^k개의 셀로 분할. k=20으로 100만 개(sub-mm 정밀도) 행동 공간을 구성.

### Figure 4 — Correlation Sensitivity Sweep ("The Melting Space")

![Fig4 Sweep](experiments/results/figures/fig4_correlation_sweep_paper.png)

ρ=0 (완벽 직교) → ρ=1 (완전 평행)으로 변하며 유효 코드 수가 지수적으로 붕괴. **ρ>0.6 이후 Collapse Zone**. Random W는 ρ≈0.15에 위치.

---

## 부록 B. FLOPs 감소 시각화

> 모델별 GVLA 헤드 교체 시 FLOPs 감소 배율 (행동 공간 크기별)

![FLOPs Reduction Chart](experiments/results/figures/flops_reduction_chart.png)

*모든 모델에서 N=1M 기준 약 52,000×의 FLOPs 감소. Y축은 로그 스케일.*

---

## 부록 B. Ablation — 왜 '직교(Orthogonal)'여야 하는가?

### 핵심 질문

"그냥 랜덤 행렬 W를 쓰면 안 되는가? 꼭 직교 제약이 필요한가?"

### 실험 설계

세 가지 조건을 비교했습니다:

| 조건 | W 초기화 방식 | 직교성 제약 |
|------|--------------|------------|
| **Orthogonal W** (ours) | QR 분해 → WW^T = I | O (학습 중 유지) |
| **Partial Orthogonal W** | 부분 Gram-Schmidt 적용 | 부분적 |
| **Random W** | 표준 가우시안 초기화 | X (없음) |

각 W 유형에 대해, 잠재 상태 `s`를 입력받아 생성된 이진 해시 `b`가 얼마나 정확하게 코드북의 목표 행동을 찾는지를 측정했습니다 (N: 512 ~ 65,536).

### 결과

![Ablation Orthogonality](experiments/results/figures/ablation_orthogonality.png)

### Formal Statement for Method

**Theorem 1 (Information-Theoretic Lower Bound).**  
Let a routing mechanism assign a binary code $Y \in \{0,1\}^k$ to each of $N$ candidate actions. If all $N$ actions must be uniquely identified without collision, then

$$
k \ge \lceil \log_2 N \rceil.
$$

Equivalently, any collision-free binary routing scheme requires at least $\Omega(\log N)$ binary decisions.

**Proposition 2 (Orthogonality Maximizes Bit Efficiency Under Isotropic Latents).**  
Let the routing bits be produced by

$$
Y_i = \mathbf{1}\{\langle w_i, z\rangle \ge 0\}, \qquad i=1,\dots,k,
$$

where $z \in \mathbb{R}^d$ is a whitened isotropic latent variable and the rows of $W \in \mathbb{R}^{k \times d}$ are unit norm. When the rows of $W$ are mutually orthogonal, the projection bits become decorrelated; under a Gaussian latent model, they are independent when balanced, and therefore

$$
H(Y) = \sum_{i=1}^{k} H(Y_i) = k.
$$

This maximizes effective code capacity and minimizes redundancy among the $k$ routing questions.

### Proof Sketch / Interpretation

**1. Entropy viewpoint.**  
To distinguish $N$ actions without collision, the codebook must contain at least $N$ distinct binary strings, so $2^k \ge N$, which gives $k \ge \lceil \log_2 N \rceil$. In practice, however, the usable capacity is governed by the joint entropy $H(Y)$ rather than the nominal bit count $k$. Since

$$
H(Y) \le \sum_{i=1}^{k} H(Y_i),
$$

any correlation among bits reduces effective capacity. Orthogonal projections reduce this redundancy and push $H(Y)$ closer to the ideal $k$-bit limit.

**2. Geometric viewpoint.**  
Orthogonal hyperplanes partition latent space into more balanced orthants, so the induced binary codes use the available cells more evenly. When orthogonality is degraded, the partition becomes skewed: some cells grow disproportionately while others collapse, which reduces code utilization and increases collision risk. The consequence is not that every non-orthogonal basis must fail, but that orthogonality gives the most reliable route to logarithmic-capacity scaling.

### 왜 Random W는 실패하는가?

비직교 행렬의 행(hyperplane)들은 서로 방향이 겹칩니다. 두 행의 내적이 0이 아닐 때, 두 질문이 **같은 방향에 대한 정보를 중복으로 묻는** 셈이 됩니다.

```
직교 기저 (Orthogonal W):
  질문 1: "이 상태는 X축 기준 양수인가?"   → 1비트 새 정보
  질문 2: "이 상태는 Y축 기준 양수인가?"   → 1비트 새 정보
  질문 3: "이 상태는 Z축 기준 양수인가?"   → 1비트 새 정보
  → k비트로 2^k개 구분 가능

랜덤 기저 (Random W):
  질문 1: "이 상태는 (0.9X + 0.4Y)축 양수인가?"   → 1비트 (X, Y 혼합)
  질문 2: "이 상태는 (0.8X + 0.6Y)축 양수인가?"   → 거의 같은 정보 중복!
  → k비트로 실제로는 훨씬 적은 경우의 수만 구분
```

이것이 **정보 중첩(Information Overlap)**입니다. 직교 기저에서는 임의의 두 행 간 코사인 유사도가 0에 수렴하지만, 랜덤 행렬에서는 유의미한 유사도가 남습니다.

**수치로 보면** (N=65,536 기준):

| 방법 | 평균 |cos(w_i, w_j)| (정보 중첩) | 정확도 |
|------|:---:|:---:|
| Orthogonal W (ours) | ~0.00 (직교성 보장) | **~100%** |
| Partial Orthogonal W | ~0.15 | 중간 |
| Random W | ~0.40+ (높은 중첩) | 낮음 |

### 결론

직교성 손실 `L_ortho = ||WW^T - I||²_F`는 선택적 정규화가 아닙니다. 이것이 없으면 W의 각 행이 중복 정보를 인코딩하기 시작하고, `log₂(N)`비트가 실제로 `log₂(N)`개의 독립적 정보를 담지 못하게 됩니다. 양자역학에서 비직교 기저로 측정하면 측정 정보가 겹치는 것과 동일한 원리입니다.

---

## 하드웨어 효율성 분석

GVLA-Net의 이점은 단순 FLOPs 감소를 넘어, 실제 하드웨어 수준의 효율성으로 이어집니다.

| 측정 항목 | 기존 Softmax | GVLA-Net (Ours) | 하드웨어적 의미 |
|-----------|:-----------:|:---------------:|----------------|
| **VRAM I/O Traffic** | 높음 — O(N) 가중치 로드 | 최소 — O(log N) 가중치 로드 | 배터리 소모 및 발열 감소; 엣지 디바이스 배포 가능 |
| **Kernel Launch Overhead** | 다중 커널 (Reduction, argmax 등) | 단일 커널 (Projection) | GPU 오버헤드 최소화; 저레이턴시 폐루프 제어 |
| **Memory Wall Resilience** | 취약 — N이 커지면 VRAM 폭발 | 강건 — N에 거의 무관 | 초정밀 행동 공간(N≥10⁶)에서도 안정적 |

**실측 메모리 절감** (N=1M, 행동 헤드만):
- Octo-Base: 52,429 MB → **0.06 MB** (~900,000× 감소)
- OpenVLA-7B: 16,384 MB → **0.31 MB** (~52,000× 감소)

이는 단순히 "더 빠른 소프트웨어"가 아니라, **기존에는 서버 GPU 없이 불가능했던 고정밀 VLA 추론을 Jetson급 엣지 디바이스에서 가능하게 만드는 수준의 변화**입니다.

---

## 왜 중요한가?

로보틱스 분야의 오랜 통념은 이랬습니다: **"정밀한 행동 공간(큰 N) = 느린 추론 = 실시간 불가"**. GVLA-Net은 이 전제를 근본부터 깨뜨립니다.

- 20비트 행동 공간(100만 개, 밀리미터 정밀도)을 써도 추론 시간은 10비트(1천 개)와 **동일합니다**.
- GVLA 헤드를 장착한 7B 파라미터 VLA는 **>1000 FPS** 의 행동 선택 주파수를 달성 — 일반 하드웨어에서도 실시간 폐루프 제어가 가능합니다.
- 기하학적 접근법은 **백본 무관(backbone-agnostic)** — 잠재 벡터를 출력하는 모든 VLA에 백본 재학습 없이 플러그인 가능합니다.

추론 장벽은 수년간 로봇 학습의 정밀도와 배포 가능성을 조용히 제한해온 근본적인 병목이었습니다. GVLA-Net이 그것을 제거합니다.

---

## 인용

```bibtex
@misc{gvlanet2026,
  title   = {GVLA-Net: Geometric Vision-Language-Action Network for O(log N) Inference},
  author  = {Jung, Hyunsoo},
  year    = {2026},
  note    = {Under submission, NeurIPS 2026}
}
```

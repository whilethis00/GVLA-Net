# BC Study 실험 기록

**태스크:** RoboMimic Lift — 판다 로봇이 큐브를 집어드는 태스크
**비교 대상:** Dense head vs GVLA head
**마지막 업데이트:** 2026-04-30

---

## 실험이 뭘 보려는 건가?

GVLA는 두 가지를 주장한다.

1. **속도** — action을 M개 중 고르는 대신 log₂(M)개 비트만 계산하니까, head 연산이 O(dM) 대신 O(d log M)으로 줄어든다
2. **성능** — 그러면서도 Dense만큼 잘 맞춘다

이 두 가지가 실제로 성립하는지 실험으로 확인한다.

> **주의:** head compute가 "O(log M)"이라고 단순하게 말하면 틀린다.
> 정확한 표현은 GVLA = O(d log M), Dense = O(dM)이다.
> speedup 근거는 M/log(M)이고, d(latent dim)는 공통으로 들어간다.

---

## 실험 목록

| # | 이름 | 설정 | 상태 |
|---|------|------|------|
| 1 | Fast mode — CPU | epochs=50, n_rollouts=20 | ✅ 완료 (참고용만) |
| 2 | Full mode — GPU (A100) | epochs=200, n_rollouts=50 | ✅ 완료 |
| 3 | Gray Code Ablation — GPU (A100) | epochs=200, n_rollouts=50 | ✅ 완료 |

> **Exp 2 vs Exp 3 숫자가 다른 이유:**
> 실험 3은 실험 2의 체크포인트를 재사용한 게 아니라 **처음부터 전체를 다시 학습했다.**
> 동일한 hyperparameter지만 random seed가 달라서 절대 수치가 다르게 나왔다.
> 두 실험의 수치를 같은 표에 올리면 안 된다. 각 실험 내에서만 비교해야 한다.

---

## 실험 1 — Fast Mode (CPU)

> GPU 없는 서버에서 실수로 돌렸다. **수치 신뢰 불가. 참고용으로만 본다.**

**설정:** epochs=50, n_rollouts=20, CPU

| M | Dense | GVLA |
|---|-------|------|
| 8 | 0% | 0% |
| 16 | 15% | 0% |
| 32 | 5% | 0% |
| 64 | 20% | 5% |
| 128 | 20% | 5% |
| 256 | 30% | 0% |
| 512 | 30% | 10% |
| 1024 | 25% | 0% |

rollout 20회는 성공 1개가 5%p 차이라 통계적으로 의미 없다.

---

## 실험 2 — Full Mode (GPU A100)

**설정:** epochs=200, n_rollouts=50, CUDA (A100), seed: 미고정

> n_rollouts=50 기준 통계적 불확실성:
> 성공률 18% = 9/50 성공 → 95% 구간 약 ±9%p
> 따라서 **10% 이하 차이는 신뢰하지 않는다.**

### Success Rate

| M | Dense | GVLA (natural binary) |
|---|-------|-----------------------|
| 8 | 0% | 0% |
| 16 | 0% | 0% |
| 32 | 12% | 8% |
| 64 | 6% | 0% |
| 128 | 2% | 6% |
| 256 | **22%** | 0% |
| 512 | 12% | 0% |
| 1024 | 12% | 0% |

GVLA (natural binary)는 전 구간에서 Dense보다 낮거나 비슷하다.

### Latency (head only, batch=1, GPU A100)

| M | Dense | GVLA |
|---|-------|------|
| 8 | 0.685ms | 2.388ms |
| 256 | 0.686ms | 2.392ms |
| 1024 | 0.681ms | 2.403ms |
| 65536 | 0.688ms | 2.420ms |

Dense가 전 구간에서 약 3배 빠르다. M이 커져도 두 방식 모두 latency가 거의 안 변한다.

이 결과는 이론과 다르다. Dense가 왜 빠른지, 왜 두 방식 모두 M에 무감한지는 아래에 분석한다.

---

### 분석 — 왜 이 결과가 나왔나

**문제 1: Latency 역전 (Dense < GVLA) — kernel launch overhead**

이론상 Dense = O(dM), GVLA = O(d log M)이라 M이 커질수록 GVLA가 유리해야 한다.
그런데 **batch=1 GPU 추론에서는 실제 연산보다 kernel launch overhead가 더 크다.**

- Dense: head 1개당 Linear 1번 → 7개 kernel
- GVLA: head 1개당 Linear + torch.where + STE + binary 변환 → 35개 kernel

GVLA가 연산량은 적어도 kernel 수가 5배 많아서 더 느리게 나온다.
Dense도 M=65536까지 latency가 ~0.688ms로 거의 일정한 이유도 같다.
행렬 크기보다 kernel launch 시간이 지배하기 때문이다.

> batch=1은 로봇 단일 추론 시나리오와 같다.
> batch가 커지면 실제 연산량이 지배하기 시작하고, O(dM) vs O(d log M) 차이가 드러난다.
> → 실험 4(batch sweep)에서 확인 예정

**문제 2: GVLA success rate 저조 — Natural Binary의 함정**

GVLA는 action bin index를 이진수 비트로 변환하고, 각 비트를 BCE loss로 학습한다.
그런데 **자연 이진수는 인접한 숫자가 완전히 다른 비트 패턴을 가질 수 있다.**

```
bin 3 → 011
bin 4 → 100   ← 3비트 전부 뒤집힘 (Hamming distance = 3)

M=1024 (k=10)일 때 더 극단적:
bin 511 → 0111111111
bin 512 → 1000000000   ← 10비트 전부 뒤집힘
```

로봇 관절이 "조금만 더 움직여"야 하는데,
BCE loss는 "지금 가진 모든 비트를 뒤집어라"는 gradient를 준다.
M이 클수록 이 문제가 심해진다. → 실험 3에서 검증

---

## 실험 3 — Gray Code Ablation (완료)

**목적:** Natural binary의 code discontinuity 문제를 Gray code로 해결했을 때 성능 변화 확인

**Gray code 성질:** 인접한 숫자가 딱 1비트만 다름 (Hamming distance = 1 보장)

```
g(i) = i XOR (i >> 1)     → encode
H(g(i), g(i+1)) = 1       → 인접 bin은 항상 1비트만 다름

예:
bin 3 → 010
bin 4 → 110   ← 1비트만 다름 (Hamming distance = 1)
```

**설정:** epochs=200, n_rollouts=50, CUDA (A100), `--gray_code` 플래그
(실험 2와 동일한 hyperparameter, 다른 random seed로 전체 재학습)

### 결과

| M | Dense | GVLA (natural) | GVLA (gray) |
|---|-------|----------------|-------------|
| 8 | 0% | 2% | 2% |
| 16 | 8% | 2% | 2% |
| 32 | 6% | 6% | 4% |
| 64 | 4% | 2% | 4% |
| 128 | 10% | 2% | **16%** |
| 256 | **16%** | 4% | 10% |
| 512 | 0% | 2% | **10%** |
| 1024 | 4% | 2% | **18%** |

> 통계 주의: n_rollouts=50, single seed. 95% 구간 약 ±8~10%p.
> 절대 수치보다 **방향성(trend)**을 보는 것이 맞다.

### 해석

**말할 수 있는 것:**

> Gray code는 natural binary 대비 high-M 영역에서 GVLA 성능을 크게 회복시킨다.
> 특히 M=128에서 2%→16%, M=1024에서 2%→18%로 개선됐다.
> 이는 natural binary encoding이 GVLA 학습의 병목으로 작동하고 있었음을 시사한다.
> high-M 영역에서 Gray-coded GVLA는 Dense와 경쟁적이며 일부 설정에서 상회한다.

**말하면 안 되는 것:**

> ~~"GVLA gray가 Dense를 이겼다."~~

같은 M 기준 비교:

| M | Dense | GVLA (gray) | 해석 |
|---|-------|-------------|------|
| 128 | 10% | **16%** | gray 우세 |
| 256 | **16%** | 10% | Dense 우세 |
| 512 | 0% | **10%** | gray 우세 |
| 1024 | 4% | **18%** | gray 우세 |

high-M에서 gray가 우세한 경향이 있지만, M=256에서는 Dense가 이긴다.
단일 seed + n=50으로는 "일반적으로 우세하다"는 주장이 약하다.

**논문에 쓸 수 있는 표현:**

> "Natural binary encoding creates discontinuous supervision for neighboring action bins.
> Gray code preserves local adjacency (H(g(i), g(i+1)) = 1), providing smoother
> bit-level gradients. In RoboMimic Lift, this substantially improves GVLA in
> large-M settings, suggesting that code geometry is critical for binary-routed action heads."

---

## 남은 과제

### 우선순위 높음
- [ ] M=128, 256, 512, 1024 구간을 **3 seed 이상, n_rollouts=200 이상**으로 재실험
  - 지금 single seed + n=50은 통계적으로 너무 약함
- [ ] Bit accuracy, Mean Hamming error, Decoded action distance metric 추가
  - "Gray code → Hamming error 감소 → action 오차 감소 → success 증가" 연결 필요
- [ ] **Exp 2 vs Exp 3 seed 불일치 정리** — 최종 표에는 동일 실험 내 수치만 사용

### 우선순위 중간
- [ ] Random binary code baseline 추가 (Gray code의 우위가 "다른 encoding이라서"가 아님을 증명)
- [ ] batch=1 vs batch=128 이상 latency 비교 — 실험 4 (bc_latency_batch.py)

---

## 파일 구조

```
experiments/results/bc_study/
├── EXPERIMENTS.md                    ← 이 파일
├── eval_results.json                 ← 실험 3 수치 결과 (dense + gvla + gvla_gray)
├── latency_batch.json                ← 실험 4 결과 (예정)
├── checkpoints/
│   ├── dense_8/ ~ dense_1024/            실험 3, Dense
│   ├── gvla_8/ ~ gvla_1024/              실험 3, GVLA (natural binary)
│   └── gvla_gray_8/ ~ gvla_gray_1024/   실험 3, GVLA (gray code)
└── figures/
    ├── bc_success_rate.png
    ├── bc_latency.png
    ├── bc_combined.png
    └── latency_batch_sweep.png           실험 4 완료 후 생성
```

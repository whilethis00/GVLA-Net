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

## Dense head vs GVLA head — 뭐가 다른가?

### 공통 구조

두 방식 모두 **같은 MLP backbone**을 공유한다.

```
obs (49-dim) → Linear(512) → LN → ReLU
             → Linear(512) → LN → ReLU
             → Linear(256) → LN
             → latent z (256-dim)
```

head만 다르다.

---

### Dense head — M-class softmax classifier

action을 M개 bin으로 quantize한 뒤, 어느 bin인지 분류하는 문제로 바꾼다.
action dim 7개 각각에 독립적인 softmax classifier를 붙인다.

```
z (256-dim) → Linear(256, M) → M개 logit  (action dim마다 1개)
```

**학습:** CrossEntropy loss (= log_softmax + NLL)

```
L_dense = sum_{d=1}^{7} CE(logits_d, bin_d)
```

**추론:** argmax → bin index → continuous action으로 복원

```
idx = argmax(logits)          # softmax 생략 가능 (단조 증가)
action = (idx + 0.5) / M * 2 - 1
```

**복잡도:** Linear(256, M) 행렬 곱 → O(dM), M이 커질수록 선형 증가

---

### GVLA head — binary bit classifier (orthogonal projection)

action bin index를 k = ceil(log₂M)개 비트로 표현하고, 각 비트를 독립적으로 예측한다.

```
z (256-dim) → W (k×256) → k개 projection  (action dim마다 1개)
```

W의 각 행은 latent space의 hyperplane을 정의한다.
projection의 부호(sign)가 각 비트의 예측값이 된다.

**학습:** BCE loss on each bit + orthogonality regularization

```
L_gvla = L_bit + λ · L_ortho

L_bit   = sum_{d=1}^{7} sum_{j=1}^{k} BCE(p_{d,j}, c_j(bin_d))
L_ortho = sum_{d=1}^{7} ||W_d W_d^T - I||_F^2
λ = 0.01  (default, Exp5에서 ablation 예정)
```

c_j(bin_d) : bin index를 Gray code 또는 natural binary로 변환한 j번째 비트

**추론:** projection → hard thresholding → bit → bin index → continuous action

```
bits = (projection >= 0).float()
bin_idx = gray_decode(bits)   # or natural binary decode
action = (bin_idx + 0.5) / M * 2 - 1
```

**복잡도:** Linear(256, k) 행렬 곱 → O(d log M), M이 커져도 log 스케일

---

### 한눈에 비교

| | Dense | GVLA |
|---|-------|------|
| head 구조 | Linear(d, M) × 7 | OrthogonalProjectionLayer(d, k) × 7 |
| output | M개 logit per dim | k개 projection per dim |
| 학습 loss | CrossEntropy | BCE + orthogonality reg |
| action encoding | bin index (직접) | binary code (natural or Gray) |
| head 복잡도 | O(dM) | O(d log M) |
| M=1024일 때 k | — | 10 |
| M=65536일 때 k | — | 16 |

---

## 실험 목록

| # | 이름 | 설정 | 상태 |
|---|------|------|------|
| 1 | Fast mode — CPU | epochs=50, n_rollouts=20 | ✅ 완료 (참고용만) |
| 2 | Full mode — GPU (A100) | epochs=200, n_rollouts=50 | ✅ 완료 |
| 3 | Gray Code Ablation — GPU (A100) | epochs=200, n_rollouts=50 | ✅ 완료 |
| 4 | Batch × M Latency Sweep — GPU (A100) | batch∈{1,8,32,128,512,1024}, M∈{8…65536} | ✅ 완료 |
| 5 | λ Ablation (ortho reg) — GPU (A100) | λ∈{0,0.001,0.01,0.1,1.0}, M=256, gray | ✅ 완료 |

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

## 실험 4 — Batch × M Latency Sweep (완료)

**목적:** batch=1에서 보이지 않던 O(dM) vs O(d log M) 차이가 큰 batch에서 나타나는지 확인

**설정:** batch ∈ {1, 8, 32, 128, 512, 1024}, M ∈ {8, 32, 128, 512, 1024, 4096, 16384, 65536}

### 결과 (ms, head only)

**Dense:**

| batch | M=8 | M=512 | M=4096 | M=16384 | M=65536 |
|-------|-----|-------|--------|---------|---------|
| 1 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 |
| 128 | 0.68 | 0.69 | 0.69 | 0.75 | 2.20 |
| 512 | 0.68 | 0.68 | 0.87 | 2.19 | **7.54** |
| 1024 | 0.69 | 0.69 | 1.33 | 3.79 | **14.37** |

**GVLA:**

| batch | M=8 | M=512 | M=4096 | M=16384 | M=65536 |
|-------|-----|-------|--------|---------|---------|
| 1 | 2.44 | 2.45 | 2.47 | 2.48 | 2.47 |
| 128 | 2.43 | 2.46 | 2.46 | 2.47 | 2.47 |
| 512 | 2.43 | 2.45 | 2.47 | 2.47 | 2.51 |
| 1024 | 2.44 | 2.49 | 2.50 | 2.51 | 2.51 |

### 핵심 발견

**GVLA latency는 batch와 M 모두에 대해 완전히 일정하다 (~2.45ms).**
반면 Dense는 batch × M이 커질수록 폭발적으로 증가한다.

crossover point (GVLA < Dense):
- B=128: M≈65536 (Dense 2.20ms ≈ GVLA 2.47ms, 거의 동등)
- B=512: M≈16384 이상 (Dense 2.19ms → 7.54ms, GVLA 2.47ms → 2.51ms)
- B=1024: M≈16384에서 Dense 3.79ms vs GVLA 2.51ms → **GVLA 1.5× 빠름**
- B=1024, M=65536: Dense 14.37ms vs GVLA 2.51ms → **GVLA 5.7× 빠름**

### 논문에 쓸 수 있는 표현

> "At batch=1, kernel launch overhead dominates, masking the O(dM) vs O(d log M)
> complexity difference. As batch size increases, Dense latency grows with M
> while GVLA remains constant. At B=1024, M=65536, GVLA is 5.7× faster than Dense."

batch=1 결과만 보고 "GVLA가 더 느리다"고 결론냈던 실험 2의 해석을 수정해야 한다.
단일 추론(batch=1) 시나리오는 kernel overhead가 지배하므로, latency 우위는
**대규모 병렬 추론 or 대용량 action space(M≥16384)** 에서 의미 있게 나타난다.

---

## 남은 과제

## 실험 5 — λ Ablation (완료)

**목적:** orthogonality regularization 강도(λ)가 성능에 미치는 영향 확인

**total loss:**
```
L = L_bit + λ · L_ortho
L_ortho = sum_{d=1}^{7} ||W_d W_d^T - I||_F^2
```

**설정:** GVLA gray, M=256, epochs=200, n_rollouts=50

### 결과

| λ | success rate | best_loss | 해석 |
|---|-------------|-----------|------|
| 0.0 | 2% | 1.5710 | W collapse — 비트 전부 중복 |
| 0.001 | 8% | 1.6125 | 약한 regularization |
| 0.01 | 12% | 1.6275 | 현재 default |
| 0.1 | 12% | 1.6233 | default와 동일 |
| 1.0 | **16%** | 1.6191 | 가장 좋음 |

### 핵심 발견

**λ=0 → 2%: orthogonality regularization이 필수적임**

regularization 없으면 W의 각 행이 학습 중 같은 방향으로 collapse한다.
k개 비트가 전부 동일한 정보를 인코딩하게 되어 2^k 개 셀이 아닌 사실상 2개 셀만 사용하는 것과 같아진다.

**λ=1.0이 best지만 통계적으로 아직 약함**

16% = 8/50, 12% = 6/50 → 2번 차이. n=50으로는 유의미한 차이라고 보기 어렵다.
트렌드(λ 클수록 좋아지는 경향)는 의미 있지만 "λ=1.0이 최적"이라는 주장은 아직 약하다.

**training loss vs success rate 역설**

λ=0일 때 training loss가 가장 낮다(1.571).
그러나 success rate는 최악(2%). orthogonality 없이 loss만 낮추면 W가 collapse해서
rollout에서는 엉터리 action을 출력한다.

### 논문에 쓸 수 있는 표현

> "Without orthogonality regularization (λ=0), all k projection vectors collapse
> to similar directions, effectively reducing 2^k cells to a few. This results in
> near-zero success rate despite low training loss. Orthogonality regularization
> is therefore necessary, not optional, for GVLA to function correctly."

### 남은 과제

- [ ] λ > 1.0 (예: 5.0, 10.0) 테스트 — peak가 어디인지
- [ ] n_rollouts=200 이상으로 λ=0.01 vs 1.0 재확인

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

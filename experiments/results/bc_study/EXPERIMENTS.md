# BC Study 실험 기록

**태스크:** RoboMimic Lift — 판다 로봇이 큐브를 집어드는 태스크
**비교 대상:** Dense head vs GVLA head
**마지막 업데이트:** 2026-04-29

---

## 실험이 뭘 보려는 건가?

GVLA의 핵심 주장은 두 가지다.

1. **속도** — action을 M개 중 하나로 고르는 대신, log₂(M)개 비트만 계산하니까 M이 커져도 추론 속도가 일정하다 (O(log M))
2. **성능** — 그러면서도 Dense만큼 잘 맞춘다

이 두 가지가 실제로 성립하는지 실험으로 확인하는 게 목표다.

---

## 실험 목록

| # | 이름 | 언제 | 상태 |
|---|------|------|------|
| 1 | Fast mode — CPU | 2026-04-29 | ✅ 완료 |
| 2 | Full mode — GPU (A100) | 2026-04-29 | ✅ 완료 |
| 3 | Gray Code Ablation — GPU (A100) | 2026-04-29 | 🔄 진행 중 |

---

## 실험 1 — Fast Mode (CPU)

> 처음에 GPU 없는 서버에서 실수로 돌렸다. 참고용 수치로만 보면 됨.

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

**주의:** rollout 20번은 너무 적어서 1~2개 차이가 5~10%p 차이로 보인다. 신뢰하기 어렵다.

---

## 실험 2 — Full Mode (GPU A100)

> 제대로 된 비교. A100에서 돌렸고, dense_8 기준 학습 38초 (CPU 대비 200배 빠름).

**설정:** epochs=200, n_rollouts=50, CUDA (A100)

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

Dense 최고: M=256에서 22% / GVLA 최고: M=32에서 8%

### Latency (head만 측정, batch=1)

| M | Dense | GVLA |
|---|-------|------|
| 8 | 0.685ms | 2.388ms |
| 256 | 0.686ms | 2.392ms |
| 1024 | 0.681ms | 2.403ms |
| 65536 | 0.688ms | 2.420ms |

Dense가 전 구간에서 3배 빠르고, 두 방식 모두 M이 커져도 latency가 거의 안 변한다.

---

### 이 결과가 왜 나왔나?

**문제 1: Latency 역전 (Dense < GVLA)**

이론상 Dense는 O(M), GVLA는 O(log M)이라 M이 커질수록 GVLA가 유리해야 한다.
그런데 batch=1 GPU 추론에서는 실제 연산보다 **GPU 커널을 띄우는 준비 비용(kernel launch overhead)**이 더 크다.

- Dense: head 1개당 Linear 연산 1번 → 7개 kernel
- GVLA: head 1개당 Linear + torch.where + STE + binary 변환 → 35개 kernel

GPU는 명령을 많이 나눌수록 느려진다. GVLA가 연산량은 적어도 명령이 5배 많아서 실제로 더 느리게 나온다.

Dense도 M=65536까지 latency가 ~0.688ms로 거의 일정한 것도 같은 이유다. 행렬 크기보다 kernel 띄우는 시간이 지배하기 때문.

**문제 2: GVLA success rate 저조 — Natural Binary의 함정**

GVLA는 action bin을 이진수 비트로 표현하고, 각 비트를 BCE loss로 학습한다.
그런데 **자연 이진수(natural binary)는 인접한 숫자가 완전히 다른 비트 패턴을 가질 수 있다**.

```
bin 3 → 011
bin 4 → 100   ← 3비트가 전부 뒤집힘 (Hamming distance = 3)
```

로봇 관절이 "조금만 더 움직여"야 하는 상황에서,
모델은 "지금 가진 모든 비트를 뒤집어라"는 엉뚱한 gradient를 받는다.
학습이 제대로 될 수가 없다.

---

## 실험 3 — Gray Code Ablation (진행 중)

> 문제 2를 고친 버전. Natural binary 대신 Gray code를 써서 성능이 올라오는지 확인한다.

**Gray code란?** 인접한 숫자가 딱 1비트만 다르게 설계된 이진수 체계.

```
bin 3 → 010
bin 4 → 110   ← 1비트만 다름 (Hamming distance = 1)
```

이걸 쓰면 "조금 더 움직여"가 "비트 1개만 바꿔"로 매핑되어 학습이 훨씬 자연스러워진다.

**설정:** epochs=200, n_rollouts=50, CUDA (A100), `--gray_code` 플래그 추가

**기대:** GVLA (gray)가 Dense 수준 이상으로 올라오면 "natural binary가 병목이었다"는 ablation 완성.

### 결과 (완료 후 채울 것)

| M | GVLA (natural) | GVLA (gray) | Dense |
|---|----------------|-------------|-------|
| 8 | 0% | TBD | 0% |
| 16 | 0% | TBD | 0% |
| 32 | 8% | TBD | 12% |
| 64 | 0% | TBD | 6% |
| 128 | 6% | TBD | 2% |
| 256 | 0% | TBD | 22% |
| 512 | 0% | TBD | 12% |
| 1024 | 0% | TBD | 12% |

---

## 앞으로 할 것

- [ ] 실험 3 결과 채우기
- [ ] batch=1 vs batch=128 latency 비교 — kernel overhead vs 실제 연산 분리
- [ ] M=65536 이상 극단 스케일에서 Dense latency 발산 확인
- [ ] 논문 ablation table 완성

---

## 파일 구조

```
experiments/results/bc_study/
├── EXPERIMENTS.md              ← 이 파일
├── eval_results.json           ← 실험 2 수치 결과 (전체)
├── checkpoints/
│   ├── dense_8/ ~ dense_1024/     실험 2, Dense
│   ├── gvla_8/ ~ gvla_1024/       실험 2, GVLA (natural binary)
│   └── gvla_gray_8/ ~ _1024/      실험 3, GVLA (gray code)  ← 진행 중
└── figures/
    ├── bc_success_rate.png
    ├── bc_latency.png
    └── bc_combined.png
```

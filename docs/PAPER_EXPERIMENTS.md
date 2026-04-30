# PAPER_EXPERIMENTS

최종 업데이트: 2026-04-30

## 목적

이 문서는 실제 로컬 실험 산출물을 기준으로, 본문에는 어떤 표만 남기고 어떤 결과를 appendix로 내릴지 정리한 논문용 실험 메모다.

핵심 원칙은 하나다.

> Main은 claim을 잠그는 표만 넣고, appendix는 투명성과 방어를 위한 표를 둔다.

## Source of Truth

아래 파일들만 수치의 최종 근거로 사용한다.

- `experiments/results/precision_custom25_200roll_core.json`
- `experiments/results/bc_study/eval_results.json`
- `experiments/results/bc_study/eval_results_2048.json`
- `experiments/results/bc_study/latency_batch.json`
- `experiments/results/bc_study/lambda_ablation/results.json`
- `experiments/results/bc_study/gray_hiconf_200roll/results.json`
- `experiments/bc_train.py`
- `experiments/bc_eval.py`
- `run_bc_study.sh`

## Main-paper 구성

본문에는 4개 표 또는 figure-table만 둔다.

1. `Table 1. Experimental setup`
2. `Table 2. Precision regime validation`
3. `Table 3. Code geometry ablation`
4. `Table 4. Scaling summary`

## Table 1. Experimental Setup

논문에서 이 표의 역할은 "무엇을 비교했고, 무엇을 공유했고, 무엇이 달랐는지"를 한 번에 고정하는 것이다.

| Category | Setup |
|---|---|
| Simulator / Robot | `robosuite` / `Panda` |
| Tasks | RoboMimic `Lift`, `custom2.5` precision placement |
| Dataset | RoboMimic Lift PH low-dim |
| Observation / Action | 49-dim state / 7D action |
| Training | Behavior cloning |
| Shared backbone | MLP, `49 -> 512 -> 512 -> 256` |
| Compared heads | Dense (`M`-way) vs GVLA (`log_2 M`-bit) |
| Evaluation | rollout success, 95% CI, head-only latency |

본문 문장:

> We use a controlled BC setting where Dense and GVLA share the same backbone and differ only in the action head, and we separately validate the need for large action resolution in a stricter precision-placement regime.

## Table 2. Precision Regime Validation

이 표는 "왜 큰 `M`이 필요한가"를 잠그는 핵심 evidence다.

**Table 2. High-resolution action spaces matter in a precision-sensitive regime.**

| Policy / Resolution | Successes | Success Rate | 95% CI | Interpretation |
|---|---:|---:|---:|---|
| Continuous | 107 / 200 | **53.5%** | 46.6-60.3% | upper reference |
| GVLA (`M=256`) | 24 / 200 | **12.0%** | 8.2-17.2% | coarse discretization collapses |
| GVLA (`M=1024`) | 86 / 200 | **43.0%** | 36.3-49.9% | strong recovery |
| GVLA (`M=2048`) | 94 / 200 | **47.0%** | 40.2-53.9% | strong recovery |

Caption draft:

> In the stricter `custom2.5` precision-placement regime, coarse discretization at `M=256` severely degrades success, while `M=1024` and `M=2048` recover much of the gap toward the continuous reference.

Safe interpretation:

- `256`은 명확히 부족하다.
- `1024/2048`은 명확히 회복된다.
- `1024`와 `2048`의 CI는 겹치므로, 둘 사이의 우열은 세게 주장하지 않는다.

## Table 3. Code Geometry Ablation

이 표는 "binary routing이 실패한 이유가 단순히 high-resolution 자체가 아니라 code geometry mismatch였는가"를 잠근다.

**Table 3. Code geometry matters for binary action routing.**

| `M` | Dense | GVLA natural | GVLA Gray |
|---:|---:|---:|---:|
| 128 | 10.0% | 2.0% | **16.0%** |
| 1024 | 4.0% | 2.0% | **18.0%** |
| 2048 | 16.0% | 4.0% | **24.0%** |

Caption draft:

> All rows are matched (`n=50`) evaluations. Natural-binary GVLA underperforms across high-resolution settings, while Gray coding substantially improves success.

Safe interpretation:

- natural binary는 high-`M`에서 consistently weak하다.
- Gray code는 `128/1024/2048`에서 natural binary보다 모두 높다.
- `M=1024`에서는 Gray가 Dense보다 뚜렷하게 높다 (`18%` vs `4%`).
- `M=2048`에서도 같은 순서가 유지된다 (`24% > 16% > 4%`).
- 다만 이 축은 single seed + `n=50` matched comparison이므로 trend 중심으로 써야 한다.

본문 문장:

> The matched sweep suggests that natural binary induces a code-geometry bottleneck, while Gray coding restores locality and substantially improves learnability in the high-resolution regime.

High-confidence rerun note:

> High-confidence reruns preserve the Gray-code signal, with `12.5%` at `M=128` and `20.5%` at `M=1024`.

## Table 4. Scaling Summary

이 표는 "GVLA가 항상 빠르다"가 아니라, "large-output regime에서 dense head의 scaling bottleneck을 제거한다"를 잠근다.

**Table 4. Head-only latency scaling under increasing batch and action resolution.**

| Batch | `M` | Dense | GVLA | Relative |
|---:|---:|---:|---:|---|
| 1 | 65536 | 0.68 ms | 2.47 ms | Dense faster |
| 128 | 65536 | 2.20 ms | 2.47 ms | near crossover |
| 512 | 65536 | 7.54 ms | 2.51 ms | GVLA 3.0x faster |
| 1024 | 16384 | 3.79 ms | 2.51 ms | GVLA 1.5x faster |
| 1024 | 65536 | 14.37 ms | 2.51 ms | GVLA 5.7x faster |

Caption draft:

> At batch `1`, GPU kernel-launch overhead dominates and Dense appears faster. As `B x M` grows, GVLA removes Dense's linear output-scaling bottleneck while remaining approximately constant in `M`.

Safe interpretation:

- batch `1`에서는 Dense가 빠르다.
- batch가 커질수록 Dense latency는 `M`과 함께 커진다.
- GVLA는 측정 범위에서 거의 constant하다.
- therefore, GVLA의 latency advantage는 large-batch or very-large-`M` regime에서 주장해야 한다.

## Method Box

Method section에는 작은 comparison box를 두는 편이 좋다.

| Head | Output per action dim | Loss | Complexity |
|---|---:|---|---|
| Dense | `M` logits | `CE(logits_j, b_j)` | `O(d h M)` |
| GVLA | `k = ceil(log2 M)` bit logits | `sum_l BCE(p_{j,l}, c_l(b_j)) + lambda L_ortho` | `O(d h log M)` |

본문 수식:

```text
L_GVLA
= sum_{j=1..D} sum_{l=1..k} BCEWithLogits(p_{j,l}, c_l(q(a_j)))
+ lambda * sum_{j=1..D} ||W_j W_j^T - I||_F^2
```

## Appendix 구성

### Appendix Table A. Full BC Sweep

| `M` | Dense | GVLA natural | GVLA Gray |
|---:|---:|---:|---:|
| 8 | 0% | 2% | 2% |
| 16 | 8% | 2% | 2% |
| 32 | 6% | 6% | 4% |
| 64 | 4% | 2% | 4% |
| 128 | 10% | 2% | 16% |
| 256 | 16% | 4% | 10% |
| 512 | 0% | 2% | 10% |
| 1024 | 4% | 2% | 18% |
| 2048 | 16% | 4% | 24% |

역할:

- Table 3의 전체 버전
- cherry-picking이 아니라는 점을 방어

주의:

- `2048` row는 `eval_results_2048.json`에서 왔다.
- 대부분 `n=50`, single seed라 절대값보다 trend로 해석해야 한다.

### Appendix Table B. Full Batch x M Latency Sweep

**Dense (ms, head only)**

| Batch | `M=8` | `M=512` | `M=4096` | `M=16384` | `M=65536` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 |
| 128 | 0.68 | 0.69 | 0.69 | 0.75 | 2.20 |
| 512 | 0.68 | 0.68 | 0.87 | 2.19 | 7.54 |
| 1024 | 0.69 | 0.69 | 1.33 | 3.79 | 14.37 |

**GVLA (ms, head only)**

| Batch | `M=8` | `M=512` | `M=4096` | `M=16384` | `M=65536` |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.44 | 2.45 | 2.47 | 2.48 | 2.47 |
| 128 | 2.43 | 2.46 | 2.46 | 2.47 | 2.47 |
| 512 | 2.43 | 2.45 | 2.47 | 2.47 | 2.51 |
| 1024 | 2.44 | 2.49 | 2.50 | 2.51 | 2.51 |

역할:

- scaling figure의 raw numbers 제공
- batch `1`에서 Dense가 빠른 이유를 숨기지 않음

### Appendix Table C. Orthogonality `lambda` Ablation

| `lambda` | Success Rate | Successes | Best Loss | Interpretation |
|---:|---:|---:|---:|---|
| 0.0 | 2% | 1 / 50 | 1.5710 | projection collapse |
| 0.001 | 8% | 4 / 50 | 1.6125 | weak regularization |
| 0.01 | 12% | 6 / 50 | 1.6275 | current default |
| 0.1 | 12% | 6 / 50 | 1.6233 | similar to default |
| 1.0 | 16% | 8 / 50 | 1.6191 | stronger regularization |
| 5.0 | **24%** | 12 / 50 | 1.6176 | best in this ablation |
| 10.0 | 10% | 5 / 50 | 1.6123 | too strong |

본문에는 한 문장만 두면 충분하다.

> Orthogonality regularization is necessary; without it, the projections collapse and rollout success drops near zero.

### Appendix Table D. High-confidence Gray reruns

| `M` | Method | Successes | Success Rate | 95% CI |
|---:|---|---:|---:|---:|
| 128 | GVLA Gray | 25 / 200 | **12.5%** | 8.6-17.8% |
| 1024 | GVLA Gray | 41 / 200 | **20.5%** | 15.5-26.6% |

역할:

- small-rollout ablation의 방향성이 유지되는지 재검증
- absolute number보다 `Gray signal survives reevaluation`를 보여주는 용도

### Appendix Table E. Full Hyperparameters

| Category | Value |
|---|---|
| Dataset | `data/robomimic/lift/ph/low_dim_v141.hdf5` |
| Observation keys | `object`, end-effector pose, gripper state, joint pos/vel variants |
| Obs dim | 49 |
| Action dim | 7 |
| Action range | clipped to `[-1, 1]` |
| Backbone | `Linear(512) -> LN -> ReLU -> Linear(512) -> LN -> ReLU -> Linear(256) -> LN` |
| Training epochs | 200 for main BC sweeps |
| Rollouts | 50 for BC sweeps, 200 for high-confidence validation |
| BC max steps | 500 |
| Precision max steps | 400 |
| Precision config | `xy_tol=0.015`, `release_clearance=0.075`, `transport_xy_thresh=0.01`, `place_height=0.068`, `kp_place=5.4` |
| Dense loss | CE |
| GVLA loss | BCE + `lambda L_ortho` |
| Default `lambda` | 0.01 |
| Hardware | A100 for BC study sweeps |

## 넣지 않는 것이 나은 것

### 1. SOTA VLA transplant

Main에는 넣지 않는 편이 안전하다.

- evidence type이 다르다
- head-replacement extrapolation 성격이 섞인다
- reviewer가 메인 story보다 이쪽을 공격할 가능성이 높다

넣더라도 appendix에서 아래 수준의 preliminary compatibility study로만 쓴다.

> Preliminary head-replacement simulations suggest compatibility with multiple VLA backbones, but we do not claim end-to-end SOTA VLA performance.

### 2. `M=4096` policy result

현재 메인에는 넣지 않는 편이 낫다.

- `1024/2048`만으로 precision-recovery claim이 충분하다
- `4096`을 넣으면 instability 설명까지 요구된다

### 3. AR latency stress / token error stress

appendix 후보는 될 수 있지만 main에는 불필요하다.

현재 main story는 아래 3축만으로 충분히 선명하다.

- precision regime
- Gray code geometry
- scaling

## 문장 가이드

강하게 써도 되는 문장:

> Some precision-sensitive regimes require substantially larger action resolution than standard easy benchmarks suggest.

> Gray coding substantially improves GVLA over natural binary in high-resolution settings.

> GVLA removes the dense head's linear output-scaling bottleneck in the large-output regime.

피해야 할 문장:

> GVLA Gray beats Dense in general.

> GVLA is always faster than Dense.

> 2048 is better than 1024.

## 현재 데이터에서 가장 안전한 one-line take-home message

> When precise control requires large action resolution, dense heads become increasingly costly, and Gray-coded GVLA offers a more scalable alternative that preserves much of the needed precision while avoiding dense linear output growth.

## 주의: 로컬 문서 간 불일치

기존 `experiments/results/bc_study/EXPERIMENTS.md`에는 `M=128` Gray 고표본 재평가가 `31/200 = 15.5%`로 적혀 있다. 하지만 현재 로컬 raw file `experiments/results/bc_study/gray_hiconf_200roll/results.json`은 다음을 기록한다.

- `gvla_gray_128 = 25/200 = 12.5%`
- `gvla_gray_1024 = 41/200 = 20.5%`

따라서 논문용 표를 만들 때는 `EXPERIMENTS.md`의 서술 문장보다 raw JSON을 우선해야 한다.

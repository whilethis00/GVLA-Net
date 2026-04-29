# NeurIPS Experiment Selection Plan

최종 업데이트: 2026-04-30

## 목적

이 문서는 현재 레포에 있는 실험 자산을 기준으로,

- 메인 본문에 들어갈 실험
- appendix로 넘길 실험
- 지금 당장 추가로 필요한 최소 실험
- 버려도 되는 실험

을 정리한 제출 직전용 문서다.

핵심 원칙은 다음과 같다.

1. 새 실험을 더 넓히지 않는다.
2. 이미 잡힌 서사를 더 안정적으로 굳힌다.
3. 메인 본문은 “왜 GVLA가 필요한가”를 가장 짧고 강하게 보여주는 실험만 남긴다.
4. appendix에는 robustness, 보조 ablation, 구현 디테일을 보낸다.

## 현재 논문 서사

지금 레포 기준으로 가장 일관된 메인 서사는 다음과 같다.

1. coarse action discretization은 실제 manipulation success를 크게 떨어뜨린다.
2. AR tokenization은 좋은 조건에서는 강하지만, decoding latency budget에 민감하다.
3. 정밀 placement regime에서는 `256 bins`가 충분하지 않을 수 있으며, `1024~2048` 수준에서야 회복되는 구간이 나타난다.
4. action resolution이 커질수록 dense / sequential decoding 비용은 빠르게 커진다.
5. structured action head는 바로 그 high-resolution regime에서 필요하다.
6. high-resolution GVLA의 학습 병목 일부는 구조 자체가 아니라 coding mismatch였고, Gray code가 이를 완화한다.

이 서사에 맞지 않는 실험은 메인에서 빼는 것이 좋다.

## 메인 본문에 들어갈 것

### Main Table Drafts

아직 결과가 덜 나온 항목도 논문 구조를 먼저 고정할 수 있도록,
아래처럼 `TBD` 형태의 표 초안을 먼저 두고 채워 넣는 방식으로 진행하는 것이 좋다.

#### Table A. Precision Regime Main Table

대상:

- task: `pick_place_can_precision`
- setting: `custom2.5`
- rollout target: `200`

| Setting | Success Rate | 95% CI | Status |
|---|---:|---:|---|
| continuous | TBD | TBD | running / fill later |
| GVLA 256 bins | TBD | TBD | running / fill later |
| GVLA 512 bins | TBD | TBD | optional |
| GVLA 1024 bins | TBD | TBD | running / fill later |
| GVLA 2048 bins | TBD | TBD | running / fill later |

권장:

- 메인 본문에서는 `continuous / 256 / 1024 / 2048`만 남기고
- `512`는 appendix 또는 supplementary inset으로 빼도 된다
- 만약 `512`가 곡선 이해에 필요하면 메인 inset에 유지

#### Table B. Gray Code Main Ablation

대상:

- same protocol
- same rollout count
- same task

| M | Dense | GVLA (natural) | GVLA (Gray) | Status |
|---|---:|---:|---:|---|
| 128 | 10% | 2% | 16% | ready |
| 1024 | 4% | 2% | 18% | ready |
| 2048 | TBD | TBD | TBD | optional / not required for main |

이 표는 지금 상태로도 메인 본문에 거의 바로 들어갈 수 있다.

#### Table C. Latency / Scaling Summary

| Measurement | Dense | GVLA | Note | Status |
|---|---:|---:|---|---|
| Head-only latency, small N | TBD | TBD | from `robosuite_study/results.json` | ready |
| Head-only latency, large N | TBD | TBD | from `robosuite_study/results.json` | ready |
| Batch=1 latency trend | TBD | TBD | from `bc_study/latency_batch.json` | ready |
| Large-batch latency trend | TBD | TBD | from `bc_study/latency_batch.json` | ready |

주의:

- 메인 표에는 raw 숫자를 전부 다 넣기보다는
- 대표 포인트 몇 개만 선택하거나
- figure 본문 + table appendix 구조로 나누는 것이 좋다

### Figure 1. 메인 메시지 그림

한 장으로 아래 세 축을 모두 보여주는 그림이 필요하다.

- coarse discretization 실패
- high resolution에서 회복
- dense/AR 비용 증가

추천 구성:

1. 왼쪽 패널: 기본 `pick_place_can` 또는 기존 Robosuite quantization 결과
   - coarse bins에서 실패
   - `96+ bins/dim`에서 회복
2. 가운데 패널: `pick_place_can_precision` (`custom2.5`)
   - `256`은 부족
   - `512`도 부족
   - `1024~2048`에서 회복
3. 오른쪽 패널: latency/scaling
   - dense vs GVLA head latency
   - large action space에서 dense cost 급증

이 한 장이 논문 메인 메시지를 거의 다 담당해야 한다.

#### Figure 1 캡션 초안

> Fine-grained action resolution can be necessary for precise manipulation, but the computational cost of conventional action heads grows rapidly with action-space size. Left: coarse discretization fails on manipulation. Center: in a stricter precision-placement regime, performance recovers only at substantially larger action resolution. Right: structured heads maintain favorable latency scaling where dense heads become prohibitively expensive.

### Figure 2. Gray Code Ablation

이건 메인에 들어갈 가치가 크다.

핵심 메시지:

- 큰 `M`에서 GVLA가 원래 안 되는 게 아니라
- natural binary encoding이 locality를 깨고 있었고
- Gray coding이 이를 완화한다

현재 이미 확인된 값:

- `dense_128 = 10%`
- `gvla_128 = 2%`
- `gvla_gray_128 = 16%`
- `dense_1024 = 4%`
- `gvla_1024 = 2%`
- `gvla_gray_1024 = 18%`

즉, 최소한 `128`과 `1024`는 메인 본문에서 충분히 강하다.

단, 원래 원했던 `2048`은 아직 없음.
따라서 메인 제출 기준으로는:

- `128`
- `1024`

두 점만으로도 충분히 강한 ablation을 만들 수 있다.

#### Figure 2 캡션 초안

> Gray coding restores locality in code space, substantially improving high-resolution GVLA performance relative to natural binary targets.

### Table 1. Precision Regime Validation

메인 표 또는 figure inset으로 넣을 것:

대상:

- `custom2.5`
- `continuous / 256 / 512 / 1024 / 2048`

현재 `10-rollout` 결과:

- `continuous = 0.6`
- `256 = 0.0`
- `512 = 0.3`
- `1024 = 0.7`
- `2048 = 0.6`
- `4096 = 0.5`

논문용으로는 `4096`보다 다음이 더 중요하다.

- `continuous`
- `256`
- `512`
- `1024`
- `2048`

이 다섯 점으로 충분하다.

다만 제출 전 반드시:

- `50`이 아니라 가능하면 `100~200 rollouts`
- confidence interval 또는 binomial CI

를 같이 넣어야 한다.

#### Table 1 템플릿

| Method / Resolution | Success Rate | 95% CI |
|---|---:|---:|
| continuous | TBD | TBD |
| GVLA 256 | TBD | TBD |
| GVLA 512 | TBD | TBD |
| GVLA 1024 | TBD | TBD |
| GVLA 2048 | TBD | TBD |

### Figure 3. Latency / Scaling

메인 본문에서는 아래 두 종류를 명확히 구분해야 한다.

1. head-only latency
2. batch scaling

현재 사용 가능한 자산:

- `experiments/results/robosuite_study/results.json`
- `experiments/results/bc_study/latency_batch.json`

권장 메시지:

- head-only: large `N`에서 dense는 급격히 느려지고 GVLA는 거의 일정
- batch scaling: batch=1과 large batch를 구분해서 latency trade-off 설명

메인 본문에서는 반드시 그림 캡션이나 본문에:

- “head-only latency”
- “end-to-end rollout latency가 아님”

을 명시해야 한다.

#### Figure 3 캡션 초안

> While task success may require increasingly fine action resolution, the cost of dense action heads scales poorly with action-space size. GVLA preserves favorable latency scaling, motivating structured action selection in large-resolution regimes.

## Appendix로 보내는 것

### Appendix Table Drafts

#### Appendix Table A. AR Latency Stress

| Setting | Continuous | AR | Status |
|---|---:|---:|---|
| 0ms | ready | ready | use existing |
| 8ms | ready | ready | use existing |
| 10ms | ready | ready | use existing |
| 15ms | ready | ready | use existing |
| 25ms | ready | ready | use existing |
| 50ms | ready | ready | use existing |
| 75ms | ready | ready | use existing |
| 100ms | ready | ready | use existing |

#### Appendix Table B. Token Error Stress

| Error Prob | Continuous | AR | Status |
|---|---:|---:|---|
| 0.02 | ready | ready | use existing |
| 0.05 | ready | ready | use existing |
| 0.10 | ready | ready | use existing |
| 0.20 | ready | ready | use existing |
| 0.30 | ready | ready | use existing |

#### Appendix Table C. Basic PickPlace Quantization Sweep

| Task / Setting | 32 | 48 | 64 | 96 | 128 | 256+ |
|---|---:|---:|---:|---:|---:|---:|
| `pick_place_can` | ready | ready | ready | ready | ready | ready |
| `pick_place_can_precision` | partial | partial | partial | partial | partial | running |

#### Appendix Table D. BC Full Sweep

| M | Dense | GVLA natural | GVLA gray |
|---|---:|---:|---:|
| 8 | ready | ready | ready |
| 16 | ready | ready | ready |
| 32 | ready | ready | ready |
| 64 | ready | ready | ready |
| 128 | ready | ready | ready |
| 256 | ready | ready | ready |
| 512 | ready | ready | ready |
| 1024 | ready | ready | ready |

### 1. AR latency stress 전체

보낼 위치:

- `experiments/results/robosuite_ar_compare_*`
- 정리 문서: `experiments/results/robosuite_ar_compare_progress.md`

이유:

- 메인 메시지에는 “AR은 좋은 조건에서는 강하지만 latency에 취약”만 있으면 충분하다.
- `0/8/10/15/25/50/75/100ms` 전체 표는 appendix가 더 적절하다.

### 2. token error 주입 결과

현재 결과:

- `0.02, 0.05, 0.10, 0.20, 0.30`까지 대부분 `100%`

이건 메인에 두면 분산만 늘어난다.
appendix에서 “token error보다 latency가 더 중요한 failure mode였다”는 보조 근거로 쓰면 충분하다.

### 3. nut assembly smoketest

현재는 baseline 자체가 죽는다.

따라서:

- 메인에서는 제외
- appendix에서도 “failed setup / non-interpretable baseline” 정도로만 언급하거나 아예 빼는 것이 낫다

### 4. orthogonality / entropy / collision / geometry 보조 그림

현재 자산:

- `experiments/results/figures/fig1_orthogonality_heatmap_paper.png`
- `fig2_bit_entropy_paper.png`
- `fig2_collision_paper.png`
- `fig3_latent_partitioning_paper.png`
- `fig4_correlation_sweep_paper.png`
- `ablation_orthogonality.*`

이것들은 Appendix용으로 매우 좋다.

메시지:

- GVLA representation geometry
- orthogonality 관련 분석
- collision / entropy 보조 분석

### 5. BC study 세부 테이블

현재 자산:

- `experiments/results/bc_study/eval_results.json`
- `experiments/results/bc_study/EXPERIMENTS.md`
- `experiments/results/appendix_tables/*.tex`

이건 appendix에 매우 적합하다.
특히 dense, natural GVLA, gray GVLA 전체 sweep 표는 appendix에서 충분히 살릴 수 있다.

## 지금 당장 필수 추가 실험

### 필수 1. custom2.5 high-confidence validation

이게 현재 가장 중요하다.

대상 포인트:

- `continuous`
- `256`
- `512`
- `1024`
- `2048`

권장 rollout 수:

- 최소 `100`
- 가능하면 `200`

메시지:

- `256`은 부족
- `512`도 완전 회복 전
- `1024~2048`에서 회복

이게 안정적으로 나오면 메인 실험 서사는 완성된다.

### 필수 2. confidence interval 계산

모든 메인 표에는 평균값만 쓰면 안 된다.

최소:

- success rate
- `95%` binomial confidence interval

을 같이 표기해야 한다.

특히 low-success regime에서는 필수다.

### 필수 3. Gray code main ablation 정리

현재 확보된 메인용 포인트:

| M | Dense | GVLA natural | GVLA gray |
|---|---:|---:|---:|
| 128 | 10% | 2% | 16% |
| 1024 | 4% | 2% | 18% |

이건 이미 메인 ablation으로 충분하다.

`2048`이 없더라도:

- `128`
- `1024`

두 점만으로 논문 메시지는 살아 있다.

### 필수 4. scaling figure 정리

현재 상태:

- `robosuite_study/results.json`은 head-only latency가 있음
- `bc_study/latency_batch.json`은 batch latency 정보가 있음
- `high_dim_action_scaling_smoke/results.json`은 차원 explosion 메시지용 초안 정도

제출 전엔 최소한 다음을 정리해야 한다.

- head-only dense vs GVLA
- batch=1 vs larger batch
- 캡션에서 측정 조건 명시

## 지금은 버릴 것

다음은 지금 욕심내지 않는 것이 맞다.

- `4096` 추가 파기
- 더 많은 precision task variants
- additional task
- ALOHA / 14D / humanoid 확장
- 2048 Gray code 추가 실험

이유:

- 메인 서사는 이미 충분히 강하다
- 지금 breadth를 늘리면 오히려 핵심 실험의 안정성이 떨어진다

즉 지금 단계에선:

> 새 실험을 넓히지 말고, 이미 있는 핵심 4개를 제출 가능한 품질로 고정하는 것이 맞다.

## 최종 메인 / Appendix 추천 구성

### Main

1. Figure 1: coarse discretization 실패 + precision regime 회복 + scaling 개요
2. Figure 2: Gray code before/after ablation
3. Table 1: `custom2.5` high-confidence success rates with CI
4. Figure 3: dense vs GVLA latency / scaling

### Appendix

1. AR latency stress 전체 표
2. token error injection 결과
3. basic pick_place sweeps 전체
4. BC full sweep table
5. orthogonality / entropy / collision / geometry 분석
6. implementation details and additional plots

## 지금 시점의 실전 우선순위

1. `custom2.5` 100~200 rollout 완성
2. 그 결과에 CI 계산
3. Gray code 128 / 1024 메인 표로 고정
4. scaling figure 한 장으로 정리
5. 그 뒤에 main/appendix figure ordering만 확정

한 줄 결론:

> 지금은 이미 “충분한 종류의 실험”은 있다.  
> 더 필요한 건 새로운 실험이 아니라, 핵심 실험 몇 개를 높은 신뢰도로 굳히고 논문 구조에 맞게 재배치하는 것이다.

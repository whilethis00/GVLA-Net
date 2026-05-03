# Experiment Matrix

## Goal

`Gray code`와 `orthogonal measurement`를 분리해서 검증한다.

현재 NeurIPS 쪽에서 이미 어느 정도 확보된 것은 `target geometry` 쪽이다.
이 트랙에서는 `measurement geometry`를 별도로 증명해야 한다.

또한 더 큰 방법론 목표는 다음 두 축을 모두 확보하는 것이다.

1. `Q-GVLA` as logarithmic binary measurement over action basis states
2. `measurement geometry` as the reason those binary measurements are non-redundant

## Minimum Variant Table

| Variant | Purpose | Status |
| --- | --- | --- |
| GVLA + Gray, no orthogonal loss | Gray only baseline | missing BC-specific split |
| GVLA + Gray + `||W W^T - I||_F^2` | parameter-space orthogonality | partially covered by `bc_lambda_ablation.py`, but mixed with `bc_study` |
| GVLA + Gray + `||Corr(WZ)-I||_F^2` | data-space decorrelation | missing |
| GVLA + Natural, no orthogonality | weak target + weak measurement geometry | partially covered |
| GVLA + Natural + orthogonality | orthogonality without Gray | missing clean comparison |
| Fixed random orthogonal `W` | learned measurement necessity | missing |
| Learned unconstrained `W` | true no-orth baseline | missing clean comparison |

## Required Diagnostics

### A. Row cosine heatmap

`C_W = W W^T`

질문:

- off-diagonal가 실제로 줄어드는가?

### B. Logit correlation heatmap

validation latent batch `Z`에 대해

`Y = Z W^T`

`Corr(Y)`

질문:

- parameter-space orthogonality가 데이터 위에서도 유지되는가?

### C. Sign-bit correlation

`b_l = 1[y_l > 0]`

질문:

- bit decisions가 실제로 덜 redundant해졌는가?

### D. Conditional entropy decay

`H(A | b_1)`

`H(A | b_1, b_2)`

`...`

질문:

- 질문 수가 늘어날수록 action uncertainty가 꾸준히 줄어드는가?
- 아니면 비슷한 질문을 반복해서 entropy가 잘 안 줄어드는가?

## Existing Reusable Experiments

### 1. Target geometry

`experiments/bc_target_bit_transition_diagnostic.py`

- natural / gray / random code의 target bit transition smoothness
- 이건 target-side geometry evidence다

### 2. Proxy measurement geometry

`experiments/orthogonality_correlation_sweep.py`

- row correlation이 커질수록 collision / occupancy가 무너지는지 확인
- measurement geometry intuition을 강화하는 proxy evidence

### 3. Trainable orthogonality ablation

`experiments/train_orthogonality_ablation.py`

- teacher hash imitation setting에서
- `ortho_coeff in {0, 1e-4, 1e-3, 1e-2}` sweep
- bit accuracy / collision / row cosine / entropy 확인 가능

## Missing BC-Specific Experiment

이 트랙에서 가장 중요한 missing piece:

`GVLA+Gray` vs `GVLA+Gray+Orth` vs `GVLA+Gray+Data-Decorrelation`

on RoboMimic Lift with:

- rollout success
- action L1 / L2
- mean `|b_hat - b|`
- Hamming error
- sign-bit correlation
- logit correlation heatmap

## Additional Longer-Horizon Experiment

시간이 충분하면 아래까지 간다.

### Product vs Entangled Q-GVLA

비교:

- Product-QGVLA (`r=1`)
- MPS-QGVLA (`r=2`)
- MPS-QGVLA (`r=4`)
- MPS-QGVLA (`r=8`)
- Dense
- autoregressive bit decoder

질문:

- product-bit independence가 충분한가?
- low-rank entangled action state가 multimodality를 더 잘 잡는가?
- 여전히 dense enumeration 없이 `O(log M)` family scaling을 유지하는가?

### Synthetic Multimodal Code Task

예시:

- valid modes only: `00000000` or `11111111`

질문:

- product model이 모든 code에 mass를 퍼뜨리는가?
- entangled / MPS model이 두 mode만 보존하는가?

## Decision Rule

- orthogonality 효과가 rollout이나 offline metric, 혹은 bit redundancy 감소에서 분명히 보이면
  - measurement geometry를 main technical pillar로 올린다
- orthogonality 효과가 약하고 Gray만 강하면
  - main contribution은 target geometry로 제한한다
  - orthogonality는 motivation / auxiliary regularization으로 내린다

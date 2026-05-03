# Q-GVLA Direction

## Core Reframing

기존 framing:

- binary action head
- Gray code
- orthogonal regularization
- quantum-inspired analogy

새 framing:

> Dense heads score all action states. Q-GVLA does not enumerate action states; it measures a logarithmic number of binary observables whose outcomes index an action state.

즉, 핵심은 `bit classifier`가 아니라 `logarithmic action measurement head`다.

## 1. Action Bins As Basis States

한 action dimension의 bin 수가 `M`일 때

`M = 2^k`

`k = log2(M)`

각 action bin `m in {0, ..., M-1}`을 `k`-bit code로 쓴다.

`c(m) in {0,1}^k`

Gray code를 쓰면 `c = g(m)`이다.

이 코드를 `k`-qubit computational basis state로 해석한다.

`|c> = |c_1 c_2 ... c_k>`

그러면 dense head는 `M`개의 basis state를 전부 score하는 방식이고,
Q-GVLA는 `k`개의 binary measurement outcome으로 basis state 하나를 찾는 방식이 된다.

## 2. Product-Qubit GVLA

policy latent:

`z = f_theta(o), z in R^h`

각 action dimension `j`에 대해 bit logit:

`eta_{j,l} = w_{j,l}^T z + beta_{j,l}`

`p_{j,l} = sigma(eta_{j,l})`

이를 qubit state로 쓰면

`|psi_{j,l}(z)> = sqrt(1 - p_{j,l}) |0> + sqrt(p_{j,l}) |1>`

그리고 전체 `k`-qubit product state는

`|Psi_j(z)> = ⊗_{l=1}^k |psi_{j,l}(z)>`

즉,

`|Psi_j(z)> = sum_{c in {0,1}^k} sqrt(P_j(c|z)) |c>`

Born probability:

`P_j(c|z) = |<c|Psi_j(z)>|^2`

product form에서는

`P_j(c|z) = Π_{l=1}^k p_{j,l}^{c_l} (1 - p_{j,l})^{1-c_l}`

정답 bin이 `m_j`, target code가 `c_j* = g(m_j)`이면

`L_Born = - sum_j log P_j(c_j* | z)`

이걸 풀면 현재 bit-wise BCE와 정확히 같다.

즉, BCE는 단순 heuristic이 아니라

> product-qubit action state에 대한 Born-rule negative log-likelihood

로 해석할 수 있다.

## 3. Complexity Claim

현실적인 per-dimension dense baseline:

`Dense: O(D h M)`

Q-GVLA:

`Q-GVLA: O(D h log M)`

joint action 기준으로 `N = M^D`이면

`K = D log2(M) = log2(N)`

이므로 joint dense exhaustive baseline과 비교하면

`O(h N) -> O(h log N)`

라고 쓸 수 있다.

단, 논문에서는 반드시

- per-dimension dense와 비교할 때는 `O(D h M) -> O(D h log M)`
- joint exhaustive와 비교할 때는 `O(h N) -> O(h log N)`

를 분리해서 써야 한다.

## 4. Information-Theoretic Lower Bound

`k`개의 binary measurement는 최대 `2^k`개의 outcome을 구분한다.

`M`개의 action bin을 식별하려면

`2^k >= M`

이어야 하므로

`k >= ceil(log2(M))`

이다.

Q-GVLA는 정확히 이 하한을 맞춘다.

논문 문장:

> Since `k` binary measurements can distinguish at most `2^k` outcomes, any binary-measurement action head requires `k >= ceil(log2(M))` measurements to identify one of `M` bins. Q-GVLA reaches this lower bound.

## 5. Measurement Geometry

각 action dimension `j`에 대해 measurement matrix:

`W_j in R^{k x h}`

`eta_j = W_j z`

기본 regularizer:

`L_basis = sum_j || W_j W_j^T - I_k ||_F^2`

하지만 실제 데이터 위에서는 latent covariance `Sigma_z` 때문에

`Cov(eta_j) = W_j Sigma_z W_j^T`

이므로 parameter-space orthogonality만으로는 충분하지 않을 수 있다.

더 강한 데이터 기반 목적식:

`L_data_meas = sum_j || Corr(W_j Z) - I_k ||_F^2`

따라서 이 트랙에서는 두 geometry를 분리한다.

- target geometry:
  - Gray-coded action basis
- measurement geometry:
  - orthogonal / decorrelated binary observables

핵심 문장:

> GVLA aligns two geometries: target geometry through Gray-coded action bases, and measurement geometry through orthogonal binary observables.

## 6. Why Orthogonality Matters

whitened latent 가정:

`z ~ N(0, I)`

measurement logits:

`eta_i = w_i^T z`

`eta_j = w_j^T z`

그러면

`Cov(eta_i, eta_j) = w_i^T w_j`

그리고 sign bit correlation은

`Corr(sign(eta_i), sign(eta_j)) = (2 / pi) arcsin(w_i^T w_j)`

따라서

`w_i^T w_j ~= 0 => bit decisions are non-redundant`

이것이 `orthogonal binary questions`를 정당화하는 핵심 수식이다.

## 7. Collapse As Entropy Reduction

물리적 wavefunction collapse를 직접 주장하지 않는다.

대신 target action code `C`와 bit measurements `B_l`에 대해

`H(C)`

`H(C | B_1, ..., B_l)`

를 본다.

좋은 measurement는 conditional information gain을 크게 만들고,
남은 action entropy를 빠르게 줄여야 한다.

즉, collapse는

> posterior entropy reduction under successive binary measurements

으로 표현한다.

## 8. Optional Extension: Entangled Q-GVLA

product-qubit model은

`P(c|z) = Π_l P(c_l | z)`

형태라서 conditional independence 가정이 강하다.

ambiguous / multimodal action에서 한계가 생기면,
tensor network / MPS 형태의 entangled action state로 확장할 수 있다.

`|Psi_j(z)> = sum_{c_1,...,c_k} alpha^T A_{j,1}^{c_1}(z) ... A_{j,k}^{c_k}(z) omega |c_1 ... c_k>`

`r = 1`이면 product model로 돌아간다.

이 확장은:

- dense enumeration 없이
- `O(log M)` scaling family를 유지하면서
- structured multimodality를 복원하는 방향이다

## 9. Locked Thesis For This Track

논문의 중심 thesis 후보:

> High-resolution robotic action prediction can be reformulated as logarithmic binary measurement over a geometry-preserving action basis.

방법 정의 문장:

> Q-GVLA is a quantum-inspired action head that represents `M` action bins as `k = log2(M)` measurement outcomes, trains the resulting product-qubit distribution with Born-rule likelihood, and regularizes the measurement axes to be non-redundant.

# Theory Note

## 1. Structural Analogy Only

이 트랙은 quantum model을 주장하지 않는다.

대신 다음의 구조적 대응만 사용한다.

| Orthogonal projective measurement | GVLA |
| --- | --- |
| state vector | policy latent `z` |
| orthogonal basis / measurement axes | row-orthonormal question vectors `w_1, ..., w_k` |
| projection | `y_l = w_l^T z` |
| measurement outcome | `b_l = 1[y_l > 0]` |
| route to one outcome | route to one binary action code |

핵심 메시지:

> binary action routing can be viewed as a sequence of latent measurements.

## 2. Core Equations

latent state `z in R^h`, measurement rows `w_l in R^h`에 대해

`y_l = w_l^T z`

`b_l = 1[y_l > 0]`

라고 두자.

두 row가 비슷하면 두 bit는 비슷한 질문을 반복한다.
즉, bit redundancy가 커진다.

### Gaussian latent assumption

`z ~ N(0, I)` 이고 `w_i`, `w_j`가 unit vector라면

`Cov(y_i, y_j) = w_i^T w_j`

따라서

`w_i^T w_j = 0`

이면 projection logits `y_i`, `y_j`는 uncorrelated이고,
Gaussian assumption 하에서는 independent다.

### Sign-bit correlation

더 중요한 식은 sign bit correlation이다.

`Corr(sign(y_i), sign(y_j)) = (2 / pi) arcsin(w_i^T w_j)`

이 식이 말해주는 것은:

- `w_i^T w_j ~= 0`
- `=> sign-bit correlation ~= 0`
- `=> different bits ask non-redundant questions`

즉, orthogonality regularization은 단순한 모양새용 제약이 아니라,
binary measurement redundancy를 줄이는 objective로 해석할 수 있다.

## 3. Parameter Space vs Data Space

기본 regularizer:

`L_orth = ||W W^T - I||_F^2`

여기서 `W in R^{k x h}`는 보통 정사각행렬이 아니므로,
정확한 표현은 `orthogonal matrix`가 아니라
`row-orthonormal measurement matrix` 또는 `row-orthogonal question matrix`다.

하지만 `W W^T = I`만으로는 충분하지 않을 수 있다.

latent covariance가 `Sigma_z`일 때 실제 데이터 위에서는

`Cov(W z) = W Sigma_z W^T`

이므로, `Sigma_z != I`이면 parameter-space orthogonality만으로
measurement output decorrelation이 보장되지 않는다.

그래서 더 강한 데이터 기반 목적식은 다음과 같다.

`L_data_orth = || Corr(W Z) - I ||_F^2`

또는

`L_cov_orth = || W Sigma_z W^T - I ||_F^2`

이 트랙에서의 핵심 구분:

- target geometry:
  - Gray code
  - neighboring bins should map to nearby codewords
- measurement geometry:
  - orthogonal / decorrelated questions
  - different bits should not ask the same thing

요약 문장:

> GVLA aligns two geometries: target geometry via Gray codes and measurement geometry via orthogonal binary questions.

## 4. What Must Be Demonstrated

이 트랙에서 orthogonality를 main contribution으로 밀려면 최소한 아래 둘 중 하나는 보여야 한다.

1. orthogonality가 bit redundancy를 줄인다
2. orthogonality가 training stability, offline metric, rollout success 중 적어도 하나를 개선한다

Gray code만 좋아지고 orthogonality 효과가 안 보이면,
main contribution은 `geometry-aware target code`로 내려가야 한다.

## 5. Connection To Q-GVLA

이 이론 메모는 `orthogonal measurement` 쪽에 집중하고,
전체 방법의 더 큰 리프레이밍은 `QGVLA_DIRECTION.md`에 정리한다.

관계는 다음과 같다.

- `QGVLA_DIRECTION.md`
  - action bin을 `k = log2(M)` measurement outcome으로 재해석
  - BCE를 Born-rule likelihood로 재해석
  - optional entangled / MPS 확장까지 포함
- `THEORY.md`
  - 왜 measurement axis가 직교 / decorrelated되어야 하는가
  - 왜 `target geometry`와 `measurement geometry`를 분리해야 하는가

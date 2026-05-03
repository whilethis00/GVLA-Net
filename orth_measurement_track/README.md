# Orthogonal Measurement Track

이 폴더는 현재 NeurIPS 제출용 결과와 분리된 별도 연구 트랙이다.

목표는 두 가지다.

1. `Gray code`가 아닌 `measurement geometry`를 main object로 세운다.
2. `양자역학 자체`를 주장하지 않고, `orthogonal projective measurement에서 영감을 받은 구조`만 방어 가능한 수준으로 정리한다.

현재 이 트랙의 더 큰 리프레이밍은 다음이다.

> Large action spaces should not be scored state-by-state. They should be measured.

즉, dense head가 `M`개의 action bin을 전부 score하는 대신,
Q-GVLA는 `k = log2(M)`개의 binary measurement outcome으로 action bin을 주소 지정한다.

## Claim Boundary

이 트랙에서 밀어야 하는 문장:

> GVLA is inspired by the structure of orthogonal projective measurement: a latent state is queried by non-overlapping measurement axes, and each measurement produces one binary decision.

피해야 하는 문장:

> GVLA applies quantum collapse to robot action prediction.

## Track Layout

- `THEORY.md`
  - 수식적 논리
  - parameter-space orthogonality vs data-space decorrelation 구분
  - `product-qubit` 해석과 Born-rule BCE 유도
- `EXPERIMENT_MATRIX.md`
  - 필요한 ablation / diagnostic / 해석 기준
- `run_measurement_geometry_track.sh`
  - NeurIPS 결과와 섞이지 않게 별도 output dir로만 실행
- `bc_measurement_geometry_ablation.py`
  - BC-specific `Gray / Natural / param-orth / data-orth / fixed-random-orth` 비교
  - offline metric + row/logit/sign correlation + entropy decay 저장
- `QGVLA_DIRECTION.md`
  - Q-GVLA 중심 서술, complexity claim, entangled extension 로드맵
- `qgvla_heads.py`
  - `ProductQGVLAHead`
  - `MPSQGVLAHead`
  - Born-style losses와 code probability helper
- `mps_multimodal_synthetic.py`
  - product vs MPS multimodal synthetic 비교

## Output Isolation

이 트랙의 결과는 아래로만 저장한다.

- `experiments/results/orth_measurement_track/`

기존 NeurIPS용 결과 폴더:

- `experiments/results/bc_study/`
- `results/`

에는 이 트랙 산출물을 섞지 않는다.

## Reusable Existing Scripts

이미 있는 스크립트 중 바로 재사용 가능한 것:

- `experiments/bc_target_bit_transition_diagnostic.py`
  - target geometry 진단
- `experiments/orthogonality_correlation_sweep.py`
  - proxy measurement correlation sweep
- `experiments/train_orthogonality_ablation.py`
  - trainable orthogonality ablation
- `orth_measurement_track/bc_measurement_geometry_ablation.py`
  - 이 트랙 전용 BC 실험 엔트리포인트

아직 별도 보강이 필요한 것:

- `GVLA+Gray+data-decorrelation` 설계의 안정화
- rollout success를 충분한 seed / rollout 수로 재확인

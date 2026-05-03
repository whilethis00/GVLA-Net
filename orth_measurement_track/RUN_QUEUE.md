# Run Queue

이 문서는 `orth_measurement_track/` 전용 실행 대기 목록이다.

원칙:

- 지금 NeurIPS 결과와 섞지 않는다.
- 출력은 항상 `experiments/results/orth_measurement_track/` 아래로만 보낸다.
- 각 항목은 짧은 설명 + 실행 명령만 남긴다.

현재 우선순위:

1. `Product-QGVLA + measurement geometry`를 먼저 굳힌다.
2. 그다음 필요하면 `entangled / MPS-QGVLA`로 확장한다.

---

## 1. Full Separate Track Suite

목적:

- target geometry, proxy measurement geometry, trainable orthogonality, BC-specific measurement geometry를 한 번에 분리 실행한다.
- 새 트랙 전체 상태를 처음 스냅샷할 때 쓰는 기본 명령이다.

명령:

```bash
bash /home/introai11/.agile/users/hsjung/projects/GVLA-Net/orth_measurement_track/run_measurement_geometry_track.sh
```

---

## 2. BC-Specific Measurement Geometry Ablation

목적:

- `Gray / Natural / param-orth / data-orth / fixed-random-orth`를 RoboMimic Lift에서 직접 비교한다.
- offline metric, row/logit/sign correlation, conditional entropy curve를 같이 저장한다.

명령:

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/orth_measurement_track/bc_measurement_geometry_ablation.py --device cuda --output-dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/orth_measurement_track/bc_measurement_geometry_ablation
```

---

## 3. BC Ablation With Rollout Success

목적:

- 위 BC ablation에 rollout success까지 같이 붙인다.
- 시간이 더 걸리지만, measurement geometry가 실제 제어 성능으로 이어지는지 보는 가장 직접적인 실행이다.

명령:

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/orth_measurement_track/bc_measurement_geometry_ablation.py --device cuda --rollouts 50 --output-dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/orth_measurement_track/bc_measurement_geometry_ablation_rollout50
```

---

## 4. Target Geometry Diagnostic Only

목적:

- natural / gray / random code의 target-bit transition smoothness만 따로 본다.
- measurement geometry와 분리해서 label/code side geometry만 빠르게 확인할 때 쓴다.

명령:

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/bc_target_bit_transition_diagnostic.py --n-bins 1024 --output-dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/orth_measurement_track/target_geometry
```

---

## 5. Proxy Measurement Correlation Sweep

목적:

- row correlation이 커질수록 collision / occupancy / reconstruction이 어떻게 무너지는지 proxy setting에서 본다.
- orthogonality intuition을 제일 싸게 강화하는 실행이다.

명령:

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/orthogonality_correlation_sweep.py --output-dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/orth_measurement_track/proxy_corr_sweep --run-name proxy_corr_sweep
```

---

## 6. Trainable Orthogonality Sweep

목적:

- teacher hash imitation setting에서 `ortho_coeff` sweep을 돌려 bit accuracy, collision, row cosine, entropy를 본다.
- BC rollout보다 싸게 “orthogonality가 redundancy를 줄이는가”를 먼저 확인할 수 있다.

명령:

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/train_orthogonality_ablation.py --device cuda --output-dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/orth_measurement_track/trainable_ortho_sweep
```

---

## 7. Product vs MPS Multimodal Synthetic

목적:

- product-bit independence가 multimodal code를 못 잡는지, 작은 bond dimension의 MPS-QGVLA가 복원하는지 본다.
- entangled 확장을 붙이기 전에 제일 싸게 보는 sanity check다.

명령:

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/orth_measurement_track/mps_multimodal_synthetic.py --device cuda --output-dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/orth_measurement_track/mps_multimodal_synthetic
```

---

## Update Rule

- 새로 돌릴 실험이 생기면 여기에 먼저 추가한다.
- 실행 후 해석은 `README.md`나 별도 결과 문서에 쓰고, 이 파일은 명령 중심으로 유지한다.

# Octo GPU Rollout Setup

이 문서는 `experiments/octo_gvla_rollout.py`를 GPU에서 재현하기 위한 현재 기준 실행 메모다.

## 기준 환경

- conda env: `octo_jax39`
- Python: `3.9`
- JAX: `0.4.20`
- JAXLIB: `0.4.20.dev20231221`
- CUDA devices 확인 완료: `cuda(id=0), cuda(id=1)`

## 왜 `octo_env`가 아니라 `octo_jax39`인가

기존 `octo_env`는 Python 3.10 + CPU JAX 조합이어서 GPU JAX 설치가 막혔다.
현재 서버/컨테이너에서는 `jaxlib` CUDA 빌드가 `python=3.9` 쪽에서만 안정적으로 풀렸다.

## 현재 런처

아래 스크립트는 `octo_jax39`를 기준으로 수정되어 있다.

- [run_octo_gvla_rollout.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/run_octo_gvla_rollout.sh)

이 스크립트는 자동으로 다음을 설정한다.

- `conda activate octo_jax39`
- `PATH=/home/introai11/.conda/envs/octo_jax39/bin:$PATH`
- `LD_LIBRARY_PATH=/home/introai11/.conda/envs/octo_jax39/lib:$LD_LIBRARY_PATH`
- `PYTHONPATH=/home/introai11/.conda/envs/octo_jax39/lib/python3.9/site-packages:...`

즉 새 세션에서도 `ptxas`를 다시 수동으로 잡을 필요가 없다.

## 기본 실행

프로젝트 루트에서:

```bash
./run_octo_gvla_rollout.sh --rollouts 10 --run-name octo_smoke
```

짧은 smoke test:

```bash
./run_octo_gvla_rollout.sh --rollouts 2 --max-steps 20 --run-name octo_smoke_short
```

## 확인된 smoke 결과

GPU short smoke:

- 결과 파일:
  [octo_gvla_rollout_summary.csv](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/octo_gvla_rollout/20260426_194505_octo_smoke/octo_gvla_rollout_summary.csv)
- `octo_coarse_256`
  `success_rate=0.000`, `mean_return=-84.055`, `mean_steps=100.00`, `mean_final_distance=0.8758`, `mean_infer_ms=65.037`
- `octo_gvla_1048576`
  `success_rate=0.000`, `mean_return=-81.645`, `mean_steps=100.00`, `mean_final_distance=0.8817`, `mean_infer_ms=53.521`

이 결과는 "파이프라인이 GPU에서 실제로 돈다"는 검증용이다. 아직 성능 결론을 내리기 위한 최종 benchmark로 해석하면 안 된다.

## 알려진 경고

아래 메시지들은 현재 기준으로 치명적 blocker가 아니다.

- `Gym has been unmaintained since 2022 ...`
- `Unable to register cuDNN/cuFFT/cuBLAS factory ...`
- `TF-TRT Warning: Could not find TensorRT`
- `Skipping observation tokenizer: obs_wrist`

## 다음 실험 권장

- rollout 수 증가: `--rollouts 20`, `--rollouts 50`
- 환경 난이도/보상 구조 조정
- `coarse_bins`, `gvla_bins` sweep
- seed 고정 후 반복 비교
- CSV를 모아 aggregate summary 추가

## 평가 원칙: 속도만으로는 부족하다

GVLA-Net의 주장이 설득력을 가지려면, 단순히 추론 시간이 줄었다는 사실만으로는 부족하다. 만약 latency가 개선되더라도 task success가 무너지면 실제 로봇 제어 관점에서는 가치가 떨어진다. 따라서 리뷰어 관점에서 더 중요한 질문은 다음이다.

- 속도가 빨라졌는가?
- 그 과정에서 success rate가 유지되거나 개선되었는가?
- reward 또는 return이 유지되거나 개선되었는가?
- tracking error / final distance / collision rate 같은 제어 품질 지표가 나빠지지 않았는가?

즉, 논문의 핵심 메시지는 단순한 `faster decoding`이 아니라 아래에 가깝다.

> GVLA-Net is only meaningful if logarithmic decoding does not come at the cost of control quality.

한국어로 풀면, `O(log N)` 디코딩이 실제 제어 품질 저하 없이 작동해야만 의미가 있다는 뜻이다.

## 따라서 꼭 보고해야 할 지표

최종 실험 표에서는 latency와 함께 아래 지표를 같이 제시하는 것이 바람직하다.

- `Success rate`
- `Mean return / episode reward`
- `Tracking error` 또는 `mean_final_distance`
- `Collision rate`
- `Inference latency`

## 실험 설계 원칙

- 같은 backbone에서 head만 바꾼 비교를 우선한다.
  - 예: `Octo coarse discrete head` vs `Octo + GVLA`
- 단순 reach 태스크만으로는 부족할 수 있다.
  - 가능하면 insertion, narrow-gap control, precise pick-place 같은 정밀 조작 태스크가 필요하다.
- 메인 주장은 아래 형태가 되어야 한다.
  - `GVLA is faster`
  - 가 아니라
  - `GVLA reduces decoding cost without collapsing task performance`

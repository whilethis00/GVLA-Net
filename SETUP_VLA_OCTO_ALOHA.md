# GVLA-Net: VLA / Octo / ALOHA Setup Notes

이 문서는 `/home/introai11/.agile/users/hsjung/projects/GVLA-Net` 기준으로,

- `vla` 환경에서 ALOHA 관련 의존성을 잡는 절차
- `Octo + ALOHA` fine-tune / eval 경로의 현재 상태
- 지금 바로 재현 가능한 경로와 아직 미검증인 경로

를 구분해서 정리한 실행 메모다.

## TL;DR

- **지금 실제로 검증된 Octo GPU 경로**는 `vla`가 아니라 `octo_jax39`다.
- 최근 업데이트 기준으로 `octo_jax39`에서 **ALOHA fine-tune -> checkpoint 저장 -> sim eval까지는 end-to-end로 실행 확인**했다.
- 다만 baseline Octo policy는 `action_horizon=5`, `window_size=1/action_horizon=1`, `window_size=2/action_horizon=1` 모두에서 **success rate 0%**였다.
- 따라서 이 문서는 “바로 논문 결과를 재현하는 완성 문서”가 아니라, **현재 repo에 들어간 ALOHA 경로를 안전하게 실행하고 디버깅하기 위한 기준 메모**로 보는 게 맞다.

## 현재 기준 환경

- Project root: `/home/introai11/.agile/users/hsjung/projects/GVLA-Net`
- `vla` env:
  - path: `/home/introai11/.conda/envs/vla`
  - Python: `3.10`
  - 용도: ALOHA / MuJoCo / Octo 관련 설치 스크립트가 이 env를 기준으로 작성됨
- `octo_jax39` env:
  - path: `/home/introai11/.conda/envs/octo_jax39`
  - Python: `3.9`
  - JAX: `0.4.20`
  - JAXLIB: `0.4.20.dev20231221`
  - 용도: **실제로 GPU JAX가 잡혀서 Octo rollout이 돌아간 검증된 환경**
- GPU:
  - A100 x2
  - Singularity 안에서 접근

관련 문서:

- [docs/OCTO_GPU_ROLLOUT_SETUP.md](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/docs/OCTO_GPU_ROLLOUT_SETUP.md)

## 상태 구분

### 1. 검증된 것

- `octo_jax39`에서 JAX GPU 사용 가능
- `run_octo_gvla_rollout.sh` 기준 Octo lightweight rollout 실행 가능
- `robot_arm_tracking_demo.py` 및 관련 figure 스크립트 실행 가능

### 2. repo에 들어갔지만 end-to-end 미검증인 것

- `scripts/setup_vla_env.sh`
- `scripts/verify_vla_env.py`
- `scripts/download_aloha_data.sh`
- `scripts/run_finetune_aloha.sh`
- `experiments/octo_aloha_eval.py`

즉, **파일은 있고 경로는 정리됐지만, ALOHA fine-tune -> checkpoint 생성 -> ALOHA sim eval 전체를 여기서 끝까지 통과시켰다고 보면 안 된다.**

## ALOHA 경로 파일들

### 설치

- [scripts/setup_vla_env.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/setup_vla_env.sh)
- [scripts/verify_vla_env.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/verify_vla_env.py)

### 데이터

- [scripts/download_aloha_data.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/download_aloha_data.sh)

### 학습

- [scripts/run_finetune_aloha.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/run_finetune_aloha.sh)

### 평가

- [experiments/octo_aloha_eval.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py)

## Step 1: `vla` 환경 의존성 설치

GPU 없이 먼저 설치:

```bash
bash /home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/setup_vla_env.sh
```

이 스크립트가 하는 일:

- `numpy==1.26.4`, `protobuf==4.25.3` 고정
- `tensorflow==2.15.0`, `tensorflow-text==2.15.0` 설치
- `jax==0.4.20`, `jaxlib==0.4.20`, `ml_dtypes==0.2.0` 재고정
- `orbax-checkpoint==0.4.4`, `flax==0.7.5`, `optax==0.1.5`, `chex==0.1.85`, `distrax==0.1.5`
- `mujoco`, `dm-control`, `gym`, `gymnasium`, `gym-aloha`
- `dlimp`
- `third_party/octo` editable install

주의:

- 이 스크립트는 **`vla` env에 CPU용 JAX를 맞추는 성격**에 가깝다.
- 여기서 `jaxlib==0.4.20`은 우리가 `octo_jax39`에서 따로 맞춘 GPU JAX 경로와는 다르다.
- 즉 설치는 가능해도, **이 자체가 곧 GPU fine-tune 보장**은 아니다.

## Step 2: 설치 검증

```bash
/home/introai11/.conda/envs/vla/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/verify_vla_env.py
```

이 스크립트가 보는 항목:

- JAX stack
  - `jax`, `jaxlib`, `flax`, `optax`, `chex`, `orbax-checkpoint`
- TensorFlow stack
  - `tensorflow`, `tensorflow_text`, `tensorflow_datasets`, `tensorflow_hub`
- Octo import
  - `octo.model`
  - `octo.model.components.tokenizers`
  - `octo.model.components.action_heads`
- ALOHA / MuJoCo
  - `mujoco`, `gym`, `gymnasium`, `dm_control`, `gym_aloha`
- `dlimp`

이 단계가 통과해야 다음 단계로 가는 게 맞다.

## Step 3: ALOHA sim 데이터 다운로드

```bash
bash /home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/download_aloha_data.sh
```

기본 다운로드 경로:

- zip: `https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip`
- extract target: `/home/introai11/.agile/users/hsjung/projects/GVLA-Net/data/aloha_sim`

기대 디렉터리:

```text
data/aloha_sim/aloha_sim_cube_scripted_dataset
```

## Step 4: ALOHA fine-tune 시도

```bash
bash /home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/run_finetune_aloha.sh \
  --pretrained_path hf://rail-berkeley/octo-small-1.5 \
  --data_dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/data/aloha_sim \
  --save_dir /home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha \
  --batch_size 128
```

중요:

- 이 래퍼는 내부적으로 `third_party/octo/examples/02_finetune_new_observation_action.py`를 호출한다.
- `--steps`는 현재 래퍼에서 실제로 소비되지 않는다.
  - 스크립트에도 `"steps are hardcoded to 5000"` 라고 적혀 있다.
- 따라서 문서에 `--steps 5000`를 적는 것보다, **현재 래퍼는 step override를 지원하지 않는다고 보는 게 정확하다.**

실행 전 확인할 것:

- Singularity 안에서 GPU가 보이는지
- `vla` env에서 JAX가 GPU를 실제로 잡는지

이 부분은 아직 **현 머신에서 완전히 검증되지 않았다.**

## Step 5: ALOHA sim eval 시도

```bash
MUJOCO_GL=egl /home/introai11/.conda/envs/vla/bin/python \
  /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py \
  --checkpoint_path /home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha/4999 \
  --rollouts 50 \
  --task AlohaInsertion-v0
```

현재 eval 스크립트 구조:

- baseline:
  - Octo의 raw continuous action 사용
- GVLA:
  - raw continuous action에 `ActionPrecisionAdapter(bins=2^20)` 후처리 적용

즉 이건 현재 **진짜 Octo head replacement라기보다 post-hoc quantization proxy**에 가깝다.  
이 점은 lightweight Octo rollout에서도 이미 중요한 해석 포인트였다.

결과 저장 위치:

```text
experiments/results/octo_aloha_eval/<timestamp>_aloha_eval.csv
```

## 현재 정확히 주의해야 할 점

### 1. `vla`는 ALOHA 스크립트용 기준 env이고, `octo_jax39`는 검증된 GPU Octo env다

이 둘을 혼동하면 안 된다.

- `vla`
  - ALOHA / MuJoCo / setup scripts 기준
- `octo_jax39`
  - 실제 GPU JAX로 Octo rollout을 돌린 기준 env

### 2. ALOHA eval은 아직 “공식 결과 생성 파이프라인”으로 보기 어렵다

이유:

- dataset download는 가능
- setup scripts도 있음
- eval script도 있음

하지만 아직:

- fine-tune checkpoint 생성
- ALOHA sim env rollout
- baseline/GVLA 성능 비교

가 이 머신에서 끝까지 검증된 상태는 아니다.

### 3. `octo_aloha_eval.py`는 현재 proxy 비교다

이 스크립트의 GVLA는:

- Octo 출력 연속 action chunk
- `ActionPrecisionAdapter`
- lattice quantization

즉 head-level replacement보다 **후처리 정밀도 adapter**에 더 가깝다.

## 추천 운영 방식

### A. 바로 뭔가 돌려야 할 때

순서:

1. `setup_vla_env.sh`
2. `verify_vla_env.py`
3. `download_aloha_data.sh`
4. `run_finetune_aloha.sh`
5. `octo_aloha_eval.py`

### B. Octo GPU 자체가 먼저 중요할 때

ALOHA보다 먼저 아래 문서를 따르는 게 안전하다:

- [docs/OCTO_GPU_ROLLOUT_SETUP.md](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/docs/OCTO_GPU_ROLLOUT_SETUP.md)

## Troubleshooting

### `ModuleNotFoundError: No module named 'jax.experimental.layout'`

`orbax-checkpoint` 버전이 너무 새로 잡힌 경우다.

```bash
/home/introai11/.conda/envs/vla/bin/pip install "orbax-checkpoint==0.4.4" --force-reinstall
```

### `ImportError: cannot import name 'LowdimObsTokenizer'`

`octo` path가 안 잡힌 경우다.

```bash
export PYTHONPATH=/home/introai11/.agile/users/hsjung/projects/GVLA-Net/third_party/octo:$PYTHONPATH
```

### `EGLError: eglBindAPI failed`

```bash
export MUJOCO_GL=egl
```

### `JAX version conflicts after pip install`

다른 패키지가 JAX stack을 덮어쓴 경우:

```bash
/home/introai11/.conda/envs/vla/bin/pip install "jax==0.4.20" "jaxlib==0.4.20" "ml_dtypes==0.2.0" --force-reinstall --no-deps
```

### `FlaxAutoModel` / `transformers` 충돌

이건 이전 Octo rollout에서도 실제로 겪은 문제다. `~/.local` user site 패키지가 먼저 잡히면 깨질 수 있다.  
가급적 env 내부 python을 직접 쓰고, 필요하면:

```bash
export PYTHONNOUSERSITE=1
```

### GPU는 보이는데 JAX가 CPU로만 도는 경우

이건 `vla` 경로보다 `octo_jax39`에서 먼저 검증된 문제다.  
ALOHA fine-tune을 진짜 GPU로 밀고 싶으면, `vla`에서의 GPU JAX 동작 여부를 별도로 다시 확인해야 한다.

## 정리

이 문서의 가장 중요한 포인트는 하나다.

- `scripts/`와 `experiments/octo_aloha_eval.py`는 **ALOHA 경로를 시작할 수 있게 정리된 상태**
- `octo_jax39` 기준으로 ALOHA fine-tune/eval 자체는 실제로 끝까지 돌릴 수 있다
- 하지만 **현재 baseline Octo ALOHA reproduction은 rollout success를 전혀 못 내므로, 이 라인은 잠정 보류 상태**다

즉 이 문서는 “완료된 ALOHA 재현 문서”가 아니라,
**현 시점에서 ALOHA fine-tune/eval을 시도할 때 덜 헷갈리게 만드는 기준 문서**다.

# ALOHA Octo Handoff

이 문서는 Claude Code나 다른 에이전트에게 현재 ALOHA 재현 상태를 넘기기 위한 인수인계 메모다.

목표는 다음이다.

- Octo finetune + ALOHA sim eval 경로를 end-to-end로 재현
- baseline Octo policy가 실제 task success를 내는지 확인
- 그 위에서 GVLA proxy 비교를 수행

현재 상태는:

- finetune은 끝까지 정상 수행됨
- checkpoint 저장도 정상
- eval 파이프라인도 실행은 됨
- 그러나 `baseline`과 `GVLA proxy` 모두 task success가 `0%`
- 따라서 지금 문제는 `GVLA가 나빠서`가 아니라 **ALOHA reproduction path 전체가 아직 안정적이지 않다**는 점이다

## 1. 현재 환경

주요 경로:

- repo root: `/home/introai11/.agile/users/hsjung/projects/GVLA-Net`
- finetune env: `octo_jax39`
- ALOHA data root:
  `/home/introai11/.agile/users/hsjung/projects/GVLA-Net/data/aloha_sim/aloha_sim_dataset/aloha_sim_cube_scripted_dataset/1.0.0`
- checkpoints:
  `/home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha`

중요:

- `vla` env는 CPU JAX가 잡히는 경우가 있었고, 최종 finetune/eval은 `octo_jax39` 기준으로 맞춤
- `octo_jax39`에서는 `jax.devices()`가 Singularity 안에서 `2`까지 보였음
- 하지만 `2GPU pmap` 경로는 첫 train step에서 XLA segfault가 나서 **1GPU fallback**으로 학습함

## 2. 현재 관련 파일

- setup/guide:
  - [SETUP_VLA_OCTO_ALOHA.md](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/SETUP_VLA_OCTO_ALOHA.md)
- env setup:
  - [scripts/setup_vla_env.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/setup_vla_env.sh)
  - [scripts/verify_vla_env.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/verify_vla_env.py)
- data download:
  - [scripts/download_aloha_data.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/download_aloha_data.sh)
- finetune wrapper:
  - [scripts/run_finetune_aloha.sh](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/run_finetune_aloha.sh)
  - [experiments/finetune_aloha_multigpu.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/finetune_aloha_multigpu.py)
- eval:
  - [experiments/octo_aloha_eval.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py)

## 3. 지금까지 해결된 문제

### 3.1 환경/의존성

`vla` 환경 충돌 해결:

- `numpy==1.26.4`
- `protobuf==4.25.3`
- `tensorflow-text==2.15.0`
- `jax==0.4.20`
- `jaxlib==0.4.20`
- `ml_dtypes==0.2.0`
- `scipy==1.11.4`
- `tensorstore==0.1.45`
- `setuptools<81`
- `einops`

실제 검증:

- TensorFlow/TFDS/TF-Text import OK
- Octo import OK
- gym_aloha / mujoco import OK

### 3.2 shell script 깨짐

다음 파일은 CRLF 문제를 고쳤음:

- `scripts/download_aloha_data.sh`
- `scripts/run_finetune_aloha.sh`

### 3.3 데이터 다운로드

실제 다운로드 완료.

압축 풀린 구조:

- `data/aloha_sim/aloha_sim_dataset/aloha_sim_cube_scripted_dataset/1.0.0`

초기 문서/스크립트는 이 경로를 정확히 가정하지 못했는데, 현재 wrapper는 자동 인식하도록 수정함.

## 4. finetune 쪽 현재 상태

### 4.1 실제 학습은 완료됨

사용 checkpoint:

- `999`
- `1999`
- `2999`
- `3999`
- `4999`

최종 checkpoint:

- [4999](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha/4999)

학습 로그 상:

- step 100: loss ~ 7.75
- step 500: loss ~ 2.12
- step 4100: loss ~ 0.48
- step 5000: loss ~ 0.42

즉 optimization 자체는 수렴하는 것처럼 보임.

### 4.2 2GPU는 불안정

초기 목표는 2GPU local data parallel이었음.

구현:

- `experiments/finetune_aloha_multigpu.py`
- `jax.pmap` 경로 추가

문제:

- 2GPU 첫 `train_step` compile/execute 시 XLA segfault
- wandb/tqdm/thread 문제로 의심했으나 제거 후에도 지속

현재 조치:

- 기본은 `--num_devices 1`
- `CUDA_VISIBLE_DEVICES=0`
- 2GPU는 experimental path로만 유지

### 4.3 finetune config의 중요한 특징

현재 finetune wrapper는:

- pretrained: `hf://rail-berkeley/octo-small-1.5`
- dataset:
  - name: `aloha_sim_cube_scripted_dataset`
  - image obs key: `top`
  - proprio key: `state`
  - language key: `language_instruction`
- observation tokenizer:
  - `wrist` 제거
  - `proprio` 추가
- action head:
  - `L1ActionHead`
  - `action_dim=14`
  - `action_horizon=50`

즉 original Octo config를 ALOHA용으로 locally rewrite하는 방식이다.

## 5. eval 쪽 현재 상태

### 5.1 `octo_aloha_eval.py`에서 고친 것

현재 eval script는 다음을 지원하도록 수정되었음.

- `checkpoint_path`가 `.../4999`처럼 step dir여도 root+step로 자동 해석
- `gym_aloha/AlohaInsertion-v0`, `gym_aloha/AlohaTransferCube-v0` alias 지원
- nested observation:
  - `pixels/top`
  - `agent_pos`
  를 Octo 입력 형식으로 변환
- `HistoryWrapper` 대신 로컬 history stack 구현
- `cv2` 제거, `PIL` resize 사용
- 디버그 옵션 추가:
  - `--debug_action_steps`
  - `--debug_save_frames`
- empirical action bounds 계산 추가
  - dataset에서 min/max를 읽어 eval action을 rescale

### 5.2 실제 env 정보 확인

`gym_aloha` 코드 확인 결과:

- env id:
  - `gym_aloha/AlohaInsertion-v0`
  - `gym_aloha/AlohaTransferCube-v0`
- observation:
  - `pixels/top`
  - `agent_pos`
- `action_space = Box(-1, 1, (14,), float32)`
- 내부 task names:
  - `insertion`
  - `transfer_cube`

### 5.3 dataset 구조 확인

TFDS에서 확인한 첫 episode/step:

- obs keys:
  - `state`
  - `top`
- `state` shape: `(14,)`
- `top` shape: `(480, 640, 3)`
- language:
  - `b'pick up the cube and hand it over'`

즉:

- observation modality 자체는 eval과 크게 안 어긋나지 않음
- language도 transfer task에 맞는 편임

## 6. 가장 중요한 실측 결과

### 6.1 action dataset 분포

첫 episode 기준 action 통계:

- shape: `(400, 14)`
- global min/max: `-1.5842 ~ 1.5827`

per-dim mean 예시:

- `[-0.0054, -0.4803, 1.0102, -0.0042, -0.5298, 1.1214, 0.5875, 0.1098, ...]`

즉 action이 단순히 `[-1, 1]` 안에 예쁘게 갇혀 있지 않다.
그래서 eval에서 empirical min/max로 다시 펼치는 로직을 추가했다.

### 6.2 debug rollout 결과

디버그 명령:

```bash
MUJOCO_GL=egl /home/introai11/.conda/envs/octo_jax39/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py --checkpoint_path /home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha/4999 --task AlohaInsertion-v0 --rollouts 1 --max_steps 50 --seed 0 --gvla_bins 1048576 --debug_action_steps 10 --debug_save_frames
```

관측:

- empirical rescale 전에는 action이 거의 `[-1, 1]` 포화
- empirical rescale 후에는 범위는 dataset-like로 맞아짐
- 하지만 rollout action이 여전히 몇몇 축에서 거의 고정
- 예:
  - `[-0.0031, -1.5842, 1.3365, 0.036..., ...]`
  - step 0~9 동안 큰 변화 없음

이건 **정책이 online feedback을 거의 못 써서 비슷한 자세를 계속 반복**하는 패턴처럼 보임.

### 6.3 eval 결과

`AlohaInsertion-v0`:

- baseline: `0%`
- GVLA proxy: `0%`

`AlohaTransferCube-v0`:

- checkpoint `999`: baseline `0%`, GVLA proxy `0%`
- checkpoint `1999`: baseline `0%`, GVLA proxy `0%`
- 이후 checkpoint도 매우 비슷한 양상으로 보임

즉:

- **GVLA만 망한 게 아니다**
- **baseline Octo finetune path 자체가 현재 성공률을 전혀 못 내고 있다**

## 7. 지금까지 배제된 가설

다음은 “유력 원인”이 아니거나, 적어도 단독 원인으로 보기 어렵다.

### 7.1 GVLA quantization만의 문제

아니다.

- baseline도 0%
- GVLA도 0%

따라서 현재 실패를 `GVLA 때문`으로 결론 내리면 안 된다.

### 7.2 observation schema mismatch만의 문제

초기엔 이게 문제였지만 현재는 많이 정리됨.

- `pixels/top`
- `agent_pos/state`
- history stacking
- prompt alias

까지는 맞춰졌다.

### 7.3 simple scaling bug만의 문제

초기엔 강한 후보였지만, empirical action bounds rescale를 넣은 뒤에도 policy quality는 살아나지 않았다.

## 8. 현재 유력 가설

여기부터가 Claude Code가 파고들어야 할 핵심이다.

### 가설 1. finetune recipe가 원본 Octo ALOHA recipe와 구조적으로 다르다

예:

- action head 교체 방식
- frozen keys
- learning rate / schedule
- dataset transform
- action_horizon/window_size
- train/eval task prompt

즉 현재 wrapper가 “대충 맞춰진 local recipe”일 뿐, 원본 strong baseline과 동등하지 않을 수 있다.

### 가설 2. eval glue code가 아직 subtle mismatch를 남기고 있다

예:

- `agent_pos == state`라 가정했지만, 실제로는 다른 representation일 수 있음
- action이 absolute target인지, current qpos와 조합해 써야 하는지
- rollout에서 첫 action만 쓰는 방식이 task에 불리할 수 있음
- ALOHA env의 reward/success semantics를 충분히 반영하지 못했을 수 있음

### 가설 3. checkpoint가 policy collapse 상태다

loss는 잘 떨어졌지만 success는 0.

즉:

- supervised loss 감소
- task success는 전혀 회복 안 됨

이는 policy collapse나 bad local optimum일 수 있다.

### 가설 4. dataset / env task mismatch

현재 dataset 이름은 `cube_scripted_dataset`.

따라서:

- `AlohaInsertion-v0`는 애초에 맞지 않을 수 있음
- `AlohaTransferCube-v0`도 기대만큼 안 되는 건, 데이터는 맞아도 recipe/eval mismatch가 더 크다는 뜻

## 9. 추천 디버깅 우선순위

Claude Code가 이 파일을 받으면 아래 순서로 보는 게 효율적이다.

### 9.1 원본 Octo recipe와 1:1 비교

해야 할 것:

- `third_party/octo/examples/02_finetune_new_observation_action.py`
- 현재 `experiments/finetune_aloha_multigpu.py`
- 둘 차이를 정확히 정리

특히:

- dataset kwargs
- traj/frame transform kwargs
- optimizer
- frozen keys
- task text
- action head definition

### 9.2 baseline-only eval sanity

GVLA 빼고:

- raw model outputs
- env_action
- qpos evolution
- reward evolution

을 first 20~50 step 동안 로그로 확인

목표:

- 로봇이 정말 안 움직이는지
- 움직이는데 틀린 방향인지
- gripper만 움직이는지
- 특정 arm만 고정되는지

### 9.3 dataset/action semantics 재검증

확인할 것:

- dataset `state`가 정확히 무엇인지
- dataset `action`이 absolute target인지 delta인지
- eval env.step(action) semantics와 truly 동일한지

### 9.4 frame-level qualitative inspection

이미 저장되는 debug frames:

- `experiments/results/octo_aloha_eval/debug_frames/...`

이걸 실제로 열어서:

- 로봇이 target 근처로 가는지
- 아예 초기 자세에서 굳어있는지
- 엉뚱한 arm이 움직이는지

를 봐야 한다.

## 10. 지금 바로 돌릴 수 있는 명령

### baseline/GVLA debug rollout

```bash
MUJOCO_GL=egl /home/introai11/.conda/envs/octo_jax39/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py --checkpoint_path /home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha/4999 --task AlohaTransferCube-v0 --rollouts 1 --max_steps 50 --seed 0 --gvla_bins 1048576 --debug_action_steps 10 --debug_save_frames
```

### checkpoint sweep

```bash
for ckpt in 999 1999 2999 3999 4999; do echo "===== CKPT $ckpt ====="; MUJOCO_GL=egl /home/introai11/.conda/envs/octo_jax39/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py --checkpoint_path /home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha/$ckpt --task AlohaTransferCube-v0 --rollouts 3 --max_steps 150 --seed 0 --gvla_bins 1048576; done
```

### env registration sanity

```bash
/home/introai11/.conda/envs/octo_jax39/bin/python -c "import gymnasium as gym, gym_aloha; env=gym.make('gym_aloha/AlohaTransferCube-v0', render_mode='rgb_array', obs_type='pixels_agent_pos'); print('action_space=', env.action_space); print('obs_space=', env.observation_space)"
```

## 11. 현재 결론

정직한 결론은 아래다.

- 지금 ALOHA 경로는 “완전 무의미”한 건 아니다.
- 실제로 finetune은 되었고, rollout도 돌고, action/obs 디버깅도 가능하다.
- 하지만 **baseline조차 0% success**이므로, 이 상태를 논문 메인 결과로 쓰면 안 된다.
- 따라서 다음 에이전트의 목표는:
  - `GVLA`를 평가하는 것 이전에
  - **baseline Octo ALOHA reproduction을 살리는 것**
  이다.

이게 살아나야 그 다음에 baseline vs GVLA 비교가 의미가 생긴다.

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

### 6.4 2026-04-28 eval script 추가 수정 사항

이 문서 초안 작성 이후 `experiments/octo_aloha_eval.py`를 더 손봤다.

핵심은 “baseline sanity check가 원본 Octo eval semantics와 다르게 돌아가고 있었다”는 점이다.

수정 내용:

- `task`별 ACT env 매핑 추가
  - `AlohaTransferCube-v0` -> `sim_transfer_cube`
  - `AlohaInsertion-v0` -> `sim_insertion`
- env reset 시 object pose sampling을 task별로 맞춤
  - transfer: cube pose
  - insertion: peg + socket pose
- language instruction도 task별로 분기
- 가장 중요:
  - 이전 로컬 eval은 model이 예측한 `50-step action chunk` 중 **첫 action 1개만 실행**하고 있었음
  - 현재는 원본 Octo eval 의도에 맞게 **predicted chunk를 그대로 순차 실행**하도록 변경함

즉, 적어도 지금의 baseline 실패는 예전처럼 “eval glue가 action chunk를 잘못 쓰고 있어서”라고 설명하기는 어려워졌다.

### 6.5 runtime 환경 우회 사항

현 머신에서는 checkpoint load 중 Orbax / etils가 파일 owner / group name lookup을 하다가 실패할 수 있었다.

증상:

- `pwd.getpwuid(): uid not found`
- `grp.getgrgid(): gid not found`

현재 `experiments/octo_aloha_eval.py`에는 이 환경에서만 필요한 lightweight fallback shim이 들어가 있다.

또한 ACT의 `utils.py`가 `torch`를 import해서, eval에서 단순 pose sampler를 쓰는 것만으로도 `torch` 의존성이 생기던 문제를 피하려고:

- cube / insertion pose sampling 함수를 eval script 내부로 로컬 복사함

이건 실험 의미를 바꾸는 수정은 아니고, 현 서버에서 eval을 실제로 끝까지 돌리기 위한 runtime workaround다.

### 6.6 최신 sanity / short-run 결과

짧은 sanity run:

- task: `AlohaTransferCube-v0`
- checkpoint: `4999`
- rollouts: `1`
- max_steps: `10`

결과:

- baseline: `0%`
- GVLA: `0%`

중요한 점:

- eval script는 이제 checkpoint load -> env 생성 -> rollout -> CSV 저장까지 정상 완료됨
- baseline 첫 chunk action은 여전히 특정 축이 강하게 고정된 패턴을 보임

### 6.7 최신 baseline repeat 결과

명령:

```bash
MUJOCO_GL=egl /home/introai11/.conda/envs/octo_jax39/bin/python /home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/octo_aloha_eval.py --checkpoint_path /home/introai11/.agile/users/hsjung/projects/GVLA-Net/checkpoints/octo_aloha/4999 --task AlohaTransferCube-v0 --rollouts 3 --max_steps 150 --seed 0 --gvla_bins 1024 --debug_action_steps 3
```

결과 CSV:

- `experiments/results/octo_aloha_eval/20260428_154714_aloha_eval.csv`

관측:

- baseline `3/3` 실패
- 각 episode 모두 `steps=150`으로 끝까지 감
- GVLA도 `3/3` 실패
- baseline 첫 chunk 시작 action 예시:
  - `[-0.0001, -1.4635, 1.1852, 0.0004, 0.3356, -0.0080, -0.0182, -0.0036, ...]`

해석:

- baseline이 조기 success를 한 번도 만들지 못함
- eval misuse를 고친 뒤에도 동일하므로, 현재 실패의 주원인을 단순 action scaling / single-step execution bug로 보기는 어려움
- 이제는 실제로
  - checkpoint quality 문제인지
  - online feedback을 거의 못 써서 fixed-like action pattern으로 무너지는지
  를 더 직접 봐야 한다

### 6.8 최신 transition-level debug 결과

`octo_aloha_eval.py`에 다음 디버그 로그를 추가했다.

- `qpos_before`
- `env_action`
- `qpos_after`
- `delta`
- `reward`
- `|dq|_mean`
- `|a-q|_mean`

짧은 확인:

- task: `AlohaTransferCube-v0`
- checkpoint: `4999`
- rollouts: `1`
- max_steps: `10`
- debug_action_steps: `3`

결과 CSV:

- `experiments/results/octo_aloha_eval/20260428_155110_aloha_eval.csv`

관측:

- baseline은 **아예 안 움직이는 것이 아니다**
- 초반 3 step에서 실제 qpos 변화가 꽤 큼
  - step 0: `|dq|_mean ~= 0.1679`
  - step 1: `|dq|_mean ~= 0.0671`
  - step 2: `|dq|_mean ~= 0.0870`
- 특히 몇몇 arm joint는 큰 폭으로 움직이지만 reward는 계속 `0.0`
- 따라서 현재 failure mode는
  - “정지”
  - 보다는 **움직이지만 task-relevant한 방향으로 수렴하지 못함**
  쪽으로 보는 게 맞다

GVLA도 마찬가지로 reward는 `0.0`이며, baseline보다 action이 더 포화된 형태를 보인다.

이 시점의 실무적 해석은:

- baseline policy가 online feedback을 잘 활용하지 못하고
- 유사한 잘못된 maneuver를 반복하고 있을 가능성이 높다

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
- ALOHA env의 reward/success semantics를 충분히 반영하지 못했을 수 있음

주의:

- `rollout에서 첫 action만 쓰는 방식`은 **이제 현재 코드 기준으로는 수정된 상태**다
- 따라서 이 항목은 “과거 유력 원인”이었고, 최신 기준에서는 이미 어느 정도 배제된 가설로 봐야 한다

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

업데이트:

- 이 우선순위는 지금도 맞다
- 다음 에이전트는 여기서 한 단계 더 나가서
  - **chunk 내부 step별 reward**
  - **qpos before/after**
  - **첫 2~3개 action chunk의 변화량**
  를 로그로 남기면 좋다

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
- eval의 큰 mismatch 하나였던 `single-action execution`은 수정됐지만, 그 뒤에도 **baseline조차 0% success**다.
- 따라서 이 상태를 논문 메인 결과로 쓰면 안 된다.
- 따라서 다음 에이전트의 목표는:
  - `GVLA`를 평가하는 것 이전에
  - **baseline Octo ALOHA reproduction을 살리는 것**
  이다.

이게 살아나야 그 다음에 baseline vs GVLA 비교가 의미가 생긴다.

## 12. 2026-04-28 추가 업데이트

이번 세션에서 `horizon`과 `window_size`를 줄여가며 추가 sanity check를 수행했다.

### 12.1 action semantics / env semantics는 큰 문제가 아님

다음을 직접 확인했다.

- dataset의 ground-truth `action`을 env에 그대로 넣으면
  `qpos_after`가 dataset의 다음 `state`와 수치적으로 거의 완전히 일치한다
- 즉 현재 ALOHA eval 경로에서:
  - action ordering
  - absolute vs delta semantics
  - gripper channel semantics
  는 큰 틀에서 맞는 것으로 봐도 된다

따라서 현재 실패를 `env.step(action)` 해석 오류로 돌리기는 어렵다.

### 12.2 `horizon=5`가 원흉은 아니었다

비교 대상:

- `octo_aloha_h1_test/499`
- `octo_aloha_h5_test/499`

offline sanity에서:

- `h5` step0 `|pred-demo| mean = 0.0972`
- `h1` step0 `|pred-demo| mean = 0.0395`

rollout에서도:

- 둘 다 `success=False`
- `h5`는 초반 qpos oscillation이 더 큼
- 따라서 `h5`를 채택할 근거는 없었음

즉 `horizon=5`가 특별히 좋아 보이지 않았고, 오히려 `h1`이 더 유망했다.

### 12.3 `action_horizon=1` 리셋도 돌파 실패

학습 스크립트에 `--window_size` 플래그를 추가했고, Orbax의 `uid 4065` 오류를 피하도록
`pwd/grp` safe patch를 넣었다.

이후 아래 두 실험을 수행했다.

#### ws1 + h1

checkpoint:

- `checkpoints/octo_aloha_ws1_h1_reset/499`

결과:

- `success_rate=0.000`
- `mean_steps=20.0`
- `mean_infer_ms=609.1`

의미:

- `multi-step horizon`만의 문제는 아니었다
- `single-frame + 1-step BC`로 줄여도 online closed-loop는 실패했다

#### ws2 + h1

checkpoint:

- `checkpoints/octo_aloha_ws2_h1_reset/499`

학습은 `batch_size=32`에서 정상 완료되었다.

결과:

- `success_rate=0.000`
- `mean_steps=20.0`
- `mean_infer_ms=766.6`

의미:

- `window_size=2`로 늘려도 rollout 성공으로 이어지지 않았다
- history를 약간 더 준다고 바로 해결되는 문제는 아니었다

### 12.4 reset mismatch도 주원인은 아닌 것으로 보임

dataset step0 이미지와 가장 비슷한 초기 화면 seed를 찾은 뒤
그 seed에서 다시 rollout을 돌려봤지만 여전히 실패했다.

즉:

- 랜덤 초기화 / 첫 시야 mismatch가 일부 영향은 줄 수 있어도
- 현재 0% success의 주원인으로 보기는 어렵다

### 12.5 현재 판단

이번 세션 기준 결론은 아래다.

- `GVLA`가 문제인 단계가 아니다
- `horizon=5`를 `1`로 줄여도 해결되지 않았다
- `window_size=1`을 `2`로 늘려도 해결되지 않았다
- action/env semantics mismatch도 주원인으로 보이지 않는다

따라서 이 라인은 잠시 보류하는 것이 맞다.

현 시점에서 ALOHA-Octo 경로는:

- 디버깅 가능한 상태이고
- finetune/eval도 끝까지 돌아가지만
- baseline policy 자체가 online rollout 성공을 만들지 못한다

즉 이 경로는 당장 논문 메인 결과를 위한 비교 라인으로 쓰기 어렵고,
필요하다면 나중에 별도 reproduction/debug task로 다시 파는 편이 맞다.

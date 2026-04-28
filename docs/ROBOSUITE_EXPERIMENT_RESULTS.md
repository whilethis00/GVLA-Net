# Robosuite 실험 결과: GVLA-Net의 조작 정밀도 필요성 검증

> **한 줄 요약:** coarse action discretization은 실제 조작에서 거의 실패한다. `PickPlaceCan`에서는 `M>=96` 수준의 fine resolution이 필요하며, 이런 full discretized grid를 Dense head로 직접 다루는 것은 비현실적이다.

---

## 왜 이 실험이 필요한가

기존 GVLA-Net 결과는 주로 head scaling과 synthetic benchmark 중심이었다.

- "헤드가 얼마나 빠른가?" → 이미 측정했다
- "실제 로봇 조작 성공률이 action resolution에 민감한가?" → 추가 증거가 필요했다

NeurIPS 리뷰어 관점에서 이 실험의 목적은 단순하다.

> "세밀한 행동 표현이 실제 manipulation success에 정말 필요한가?"

이 Robosuite 실험은 그 질문에 직접 답한다.

---

## 실험 설계

핵심 아이디어는 학습 자체를 비교하는 것이 아니라, **동일한 연속 expert action을 얼마나 거칠게 양자화하면 조작이 무너지는지**를 보는 것이다.

- 환경: Robosuite `PickPlace` with `Panda`
- 태스크: `single_object_mode=2`, `object_type="can"` (`PickPlaceCan`)
- 컨트롤러: OSC delta pose
- 행동 차원: 7 (`Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper`)
- 정책: scripted continuous policy
- Quantization:
  - motion 6차원은 symmetric nearest-bin quantization
  - gripper는 binary open / close
- 평가: 100 rollouts per condition, 400 steps per rollout

이 설계에서 policy는 고정되어 있고, 바뀌는 것은 오직 action resolution `M`뿐이다.
따라서 성공률 변화는 학습 noise가 아니라 quantization error의 영향으로 해석할 수 있다.

---

## 태스크 선택 이유

두 개의 태스크를 비교했다.

1. `Lift`
- 쉬운 pick-up task
- `M=32`에서 이미 빠르게 포화
- resolution effect의 존재는 보여주지만, precision bottleneck을 강하게 드러내지는 못함

2. `PickPlaceCan`
- 집기 + 목표 bin으로 이동 + 정확한 배치까지 필요
- `Lift`보다 훨씬 precision-sensitive
- action resolution threshold를 더 분명하게 보여줌

결론적으로, 논문용 메인 조작 결과는 `PickPlaceCan`을 중심으로 제시하는 것이 적절하다.

---

## 결과 1: PickPlaceCan 성공률 vs action resolution

100 rollouts 기준:

| M (bins / dim) | Full discretized grid equivalent `N_total = M^7` | 성공률 |
|:---:|---:|:---:|
| Continuous | ∞ | **100%** |
| 32 | 34,359,738,368 | **1%** |
| 48 | 587,068,342,272 | **17%** |
| 64 | 4,398,046,511,104 | **81%** |
| 96 | 75,144,747,810,816 | **100%** |
| 128 | 562,949,953,421,312 | **100%** |

### 해석

- `M=32`에서는 거의 항상 실패한다.
- `M=48`에서도 대부분 실패한다.
- `M=64`에서야 성공률이 크게 회복된다.
- `M>=96`에서 continuous expert와 동일한 `100%` 성공률에 도달한다.

즉, `PickPlaceCan`에서는 coarse discretization으로는 manipulation이 사실상 불가능하며, **fine action resolution이 실제 success의 전제 조건**이다.

---

## 결과 2: Lift는 빠르게 포화, PickPlaceCan은 높은 임계 정밀도를 요구

이전 `Lift` 결과:

| M (bins / dim) | 성공률 |
|:---:|:---:|
| Continuous | 100% |
| 4 | 54% |
| 8 | 70% |
| 16 | 88% |
| 32 | 100% |

비교하면:

- `Lift`는 상대적으로 쉬운 태스크라 `M=32`에서 이미 포화된다.
- `PickPlaceCan`은 `M=32`에서 `1%`에 불과하고, `M=96` 근처에서야 포화된다.

이 대비는 중요한 메시지를 준다.

> 필요한 action resolution threshold는 task-dependent이며, 정밀한 manipulation일수록 훨씬 더 fine한 discretization이 필요하다.

---

## 결과 3: Dense head vs GVLA head

`PickPlaceCan`에서 saturation이 시작되는 `M=96`을 full discretized 7-D grid equivalent로 보면:

- `N_total = 96^7 = 75,144,747,810,816`
- `k = ceil(log2 N_total) = 47`

Dense head와 GVLA head를 비교하면:

| 항목 | Dense Head | GVLA Head |
|------|:----------:|:---------:|
| 파라미터 수 | `N_total × d` | `k × d` |
| 메모리 스케일링 | `O(N)` | `O(log N)` |
| 추론 스케일링 | `O(N)` | `O(log N)` |
| `N_total = 96^7`에서 직접 저장 가능? | ✗ | ✓ |

중요한 점은, 여기서의 `N_total = M^7`은 **full ambient 7-D uniform grid equivalent**라는 것이다.

- scripted policy는 orientation 3축을 거의 사용하지 않는다
- gripper도 사실상 binary이다

따라서 이 수치를 "태스크의 intrinsic action count"로 직접 해석하면 안 된다.
하지만 **dense head가 full discretized action grid를 직접 다루는 방식이 얼마나 빨리 비현실적이 되는지**를 보여주는 지표로는 충분히 유효하다.

---

## 논문 contribution으로서의 의미

### 리뷰어 질문

> "이게 실제 manipulation에서 의미 있는가?"

### 이 실험의 답

1. **Manipulation relevance**
- `PickPlaceCan`에서 action resolution이 실제 task success를 결정한다.
- `M=32`는 거의 실패, `M=96`은 100% 성공이다.

2. **Task-dependent precision requirement**
- 쉬운 `Lift`에서는 `M=32`면 충분하다.
- 더 어려운 `PickPlaceCan`에서는 훨씬 더 높은 resolution threshold가 필요하다.

3. **System relevance**
- 조작 성공에 필요한 resolution이 높아질수록 dense action head는 저장과 추론 양쪽에서 빠르게 비현실적이 된다.
- GVLA는 같은 full discretized grid equivalent를 `O(log N)`으로 다룰 수 있다.

4. **Backbone-agnostic transferability**
- 이 Robosuite 실험은 action-resolution requirement 자체를 보여준다.
- GVLA head는 별도의 실험들에서 Octo, OpenVLA, RT-2-X, pi0.5 등에 이식 가능함을 보였다.

---

## 한계와 정직한 서술

이 실험은 강하지만, 과장해서 쓰면 공격받는다. 안전한 서술은 아래와 같다.

- 이 실험은 **VLA backbone 성능 비교**가 아니다.
- scripted continuous policy를 고정한 상태에서 **action discretization threshold**를 측정한 것이다.
- `N_total = M^7`은 intrinsic complexity가 아니라 **full discretized grid equivalent**이다.
- 따라서 가장 정확한 메시지는:

> practical manipulation success can require substantially finer action resolution than coarse dense heads can efficiently support over the corresponding full discretized action grid.

---

## 재현 방법

```bash
source /tools/anaconda3/etc/profile.d/conda.sh
conda activate dhmamba

cd /path/to/GVLA-Net

python experiments/robosuite_quantization_study.py \
    --task pick_place_can \
    --n_rollouts 100 \
    --max_steps 400 \
    --device cpu \
    --skip_latency \
    --save_dir experiments/results/robosuite_pickplace_transition_100roll
```

---

## 핵심 메시지

> **In PickPlaceCan, success rises from 1% at `M=32` to 100% at `M>=96`. Fine action resolution is not an efficiency luxury; it is a prerequisite for reliable manipulation.**

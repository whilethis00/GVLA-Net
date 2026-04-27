# GVLA-Net 교수님 설득용 PPT 원고

이 문서는 PPT에 바로 옮기기 쉽게 다시 쓴 버전이다.

- 슬라이드 제목
- 슬라이드에 넣을 짧은 문구
- 발표할 때 말할 멘트
- 실험 배경 / 세팅 / 결과 / 해석
- 넣을 표 / figure

전체 톤은 논문 문장보다 발표 문장에 맞춘다.

---

## Slide 1. Title

### 제목

`GVLA-Net: Geometric Vision-Language-Action Network`

### 소제목

`Breaking the Action Inference Wall from O(N) to O(log N)`

### 슬라이드에 넣을 말

- action을 전부 비교하지 않고 찾을 수 없을까?
- large action space에서 더 가벼운 routing이 가능할까?
- GVLA는 이 질문에서 출발

### 발표 멘트

이 연구는 action을 더 빨리 뽑는 팁 하나를 붙이는 얘기가 아닙니다.  
지금처럼 action 후보를 다 비교하는 대신, 더 적은 수의 구조적인 질문으로 action을 찾을 수 있는지 보자는 이야기입니다.

---

## Slide 2. Problem

### 제목

`왜 지금 action head를 다시 봐야 하나`

### 소제목

`backbone은 커졌는데, 마지막 action interface는 아직 무겁다`

### 슬라이드에 넣을 말

- backbone은 계속 강해짐
- 그런데 action head는 여전히 후보를 많이 봐야 함
- action space가 커질수록 latency / memory 부담도 같이 커짐

### 발표 멘트

요즘 VLA는 backbone 쪽은 정말 빠르게 좋아지고 있습니다.  
그런데 마지막 action selection은 아직도 큰 후보 집합을 직접 다루는 경우가 많아서, action space가 커질수록 병목이 생길 수 있습니다.

---

## Slide 3. Current Direction

### 제목

`요즘 VLA 흐름에서 우리가 건드리고 싶은 지점`

### 소제목

`학습을 더 잘하는 문제보다, 추론을 더 싸게 만드는 문제`

### 슬라이드에 넣을 말

- 요즘 VLA는 크게 세 방향
- 더 큰 backbone
- 더 나은 action representation
- 더 효율적인 inference / deployment

### 발표 멘트

GVLA는 backbone을 바꾸려는 게 아닙니다.  
학습을 무조건 더 잘하게 만들겠다는 주장도 아닙니다.  
저희가 보는 포인트는 action interface, 그중에서도 추론 시점의 routing cost입니다.

---

## Slide 4. What We Replace

### 제목

`우리가 바꾸려는 것 / 안 바꾸려는 것`

### 소제목

`VLA 전체가 아니라 action inference interface`

### 슬라이드에 넣을 말

#### 바꾸려는 것

- dense action scoring
- 큰 discrete action dictionary를 직접 비교하는 구조
- 선형 탐색형 action routing

#### 안 바꾸려는 것

- visual encoder
- language backbone
- planner / memory / world model 전체
- 데이터셋 자체

### 발표 멘트

이건 OpenVLA나 Octo 전체를 갈아엎자는 얘기가 아닙니다.  
조금 더 정확히 말하면, backbone이 latent를 만든 뒤 마지막에 action으로 보내는 그 인터페이스를 다시 설계해보자는 제안입니다.

---

## Slide 5. Scope

### 제목

`주장 범위는 어디까지인가`

### 소제목

`learning improvement라기보다 inference improvement`

### 슬라이드에 넣을 말

#### 직접 겨냥

- large action regime에서 latency
- action head memory cost
- dense enumeration의 구조적 비효율

#### 직접 주장 안 함

- 학습이 항상 더 잘 됨
- 모든 continuous policy를 대체
- 모든 태스크 성공률이 자동으로 상승

### 발표 멘트

이 부분은 현실적으로 말하는 게 중요합니다.  
저희는 학습이 무조건 좋아진다고 말하고 싶지 않습니다.  
대신 large action space에서 추론 구조를 더 가볍게 만들 수 있는지에 집중하고 있습니다.

---

## Slide 6. Relation to Diffusion / Flow Matching

### 제목

`diffusion / flow matching이랑 무슨 관계인가`

### 소제목

`정면 대체라기보다, action interface 쪽의 다른 선택지`

### 슬라이드에 넣을 말

- dense classification head는 직접 비교 대상
- token-level action routing도 직접 비교 대상
- diffusion / flow matching은 완전한 1:1 대체 관계라고 보기 어려움
- 다만 action interface를 다시 설계하는 관점에서는 비교 가치가 있음

### 발표 멘트

특히 flow matching은 조심해서 말해야 합니다.  
저희가 flow matching 전체를 대체한다고 말하는 건 무리입니다.  
더 정확한 표현은, action을 어떻게 parameterize하고 어떻게 추론할지에 대해 다른 방향을 제안한다는 쪽입니다.

---

## Slide 7. Why Softmax Becomes Heavy

### 제목

`왜 기존 방식이 무거워지는가`

### 소제목

`후보를 전부 보는 구조라서`

### 슬라이드에 넣을 말

```text
score_i = <s, a_i>,  i = 1, ..., N
action = argmax_i score_i
```

- 후보 수 `N`만큼 비교
- 메모리도 `N`에 따라 커짐
- action space가 커질수록 선형적으로 부담 증가

### 발표 멘트

softmax가 문제라기보다, 모든 후보를 직접 보는 구조가 문제입니다.  
후보가 작을 때는 괜찮지만, 3만 개, 100만 개로 가면 이 부분이 점점 무거워집니다.

---

## Slide 8. Quantum Motivation

### 제목

`양자역학에서 가져온 직관`

### 소제목

`직교한 관측은 중복이 적다`

### 슬라이드에 넣을 말

- orthogonal measurement
- 겹치지 않는 정보
- 독립적인 질문
- action routing에 이 직관을 가져오기

### 발표 멘트

양자역학에서 공식을 그대로 가져온 건 아닙니다.  
저희가 가져온 건 직교한 관측은 서로 중복이 적다는 직관입니다.  
이걸 latent action space에 옮기면, 각 projection이 서로 다른 질문 역할을 할 수 있습니다.

---

## Slide 9. Core Idea

### 제목

`핵심 아이디어`

### 소제목

`다 점수 매기지 말고, 질문으로 좁혀 가기`

### 슬라이드에 넣을 말

- 기존: 모든 action 후보를 직접 점수 매김
- GVLA: `k = ceil(log2 N)`개의 binary question
- 각 질문의 답을 bit로 모아서 action code 생성

### 발표 멘트

핵심은 scoring에서 routing으로 바꾸는 겁니다.  
예전에는 전부 비교했다면, 이제는 몇 개의 yes/no 질문으로 후보를 점점 좁혀 가는 방식입니다.

---

## Slide 10. Method

### 제목

`GVLA 수식`

### 소제목

`학습 가능한 orthogonal projection layer`

### 슬라이드에 넣을 말

```text
s in R^d
W in R^(k x d),  k = ceil(log2 N)
y = W s
b = sign(y)
L_ortho = ||W W^T - I||_F^2
```

- `s`: latent state
- `W`: 질문 방향
- `b`: binary routing code

### 발표 멘트

수식은 생각보다 단순합니다.  
latent를 몇 개의 방향으로 투영하고, 부호를 보면서 bit를 만듭니다.  
여기서 중요한 건 이 방향들이 서로 최대한 겹치지 않게 학습된다는 점입니다.

### 코드 근거

- [models/layers.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/models/layers.py)
- [utils/geometry.py](/home/introai11/.agile/users/hsjung/projects/GVLA-Net/utils/geometry.py)

---

## Slide 11. Why Orthogonality

### 제목

`왜 굳이 직교여야 하나`

### 소제목

`질문끼리 비슷하면 bit 수만 많고 정보는 안 늘어남`

### 슬라이드에 넣을 말

- 비슷한 질문 = 중복된 bit
- 중복된 bit = 낮은 code capacity
- orthogonality = 각 bit가 다른 정보 담당

### 발표 멘트

질문들이 서로 비슷하면 bit를 여러 개 써도 실제로는 같은 얘기를 반복하게 됩니다.  
그래서 orthogonality는 예쁘게 보이기 위한 제약이 아니라, bit를 제대로 쓰기 위한 핵심 조건입니다.

---

## Slide 12. Why log N

### 제목

`왜 O(log N)이 나오나`

### 소제목

`N개를 구분하는 데 필요한 yes/no 질문 수`

### 슬라이드에 넣을 말

```text
2^k >= N
```

```text
k = ceil(log2 N)
```

- `k`개의 binary decision
- 최대 `2^k`개 구분 가능
- 그래서 필요한 질문 수는 `log2 N`

### 발표 멘트

여기서 중요한 건 구현 트릭이 아닙니다.  
핵심은 `N`개를 구분하려면 yes/no 질문이 몇 개 필요한가입니다.  
그 관점으로 보면 자연스럽게 `log N`이 나옵니다.

---

## Slide 13. Proposition

### 제목

`발표용 proposition`

### 소제목

`중복 없는 binary routing이면 log2(N)개의 질문으로 충분`

### 슬라이드에 넣을 말

**Proposition**

- action마다 다른 binary code가 있고
- 각 bit가 충분히 중복되지 않는다면
- `N`개 action을 식별하는 데 필요한 질문 수는 `ceil(log2 N)`

### 발표 멘트

발표에서는 이 정도 proposition으로 충분하다고 생각합니다.  
핵심은 `왜 log N이 나오는지`와 `왜 orthogonality가 필요한지`를 같이 묶어서 보여주는 것입니다.

---

## Slide 14. Experimental Map

### 제목

`실험에서 보여주고 싶은 것`

### 소제목

`속도만이 아니라 구조와 해석까지`

### 슬라이드에 넣을 말

| 실험 축 | 보고 싶은 질문 |
| --- | --- |
| Scaling | action space가 커질수록 진짜 유리한가 |
| Transplant | 여러 backbone에서도 통하는가 |
| Orthogonality ablation | 직교성이 진짜 핵심인가 |
| Entropy / correlation | 내부 메커니즘이 보이는가 |
| Tracking | control 쪽에서도 의미가 있는가 |

### 발표 멘트

실험은 하나만 잘 나오면 부족하다고 생각했습니다.  
그래서 속도, 구조, 메커니즘, 응용성, 범용성을 따로 나눠서 봤습니다.

---

## Slide 15. Experiment 1 Background

### 제목

`Experiment 1. Scaling`

### 소제목

`action space가 커질수록 gap이 어떻게 변하나`

### 슬라이드에 넣을 말

#### 배경

- GVLA의 핵심 가설은 large action regime에서 더 잘 드러남

#### 세팅

- `N = 1,024 / 32,768 / 1,048,576`
- dense head vs GVLA head latency 비교

### 발표 멘트

이 실험은 절대 시간 자랑이 목적은 아닙니다.  
핵심은 `N`이 커질수록 두 방식의 scaling 모양이 얼마나 다르게 보이느냐입니다.

---

## Slide 16. Experiment 1 Result

### 제목

`Experiment 1. Scaling Result`

### 소제목

`N이 커질수록 GVLA 쪽 advantage가 커짐`

### figure

![Pareto Efficiency Figure](../experiments/results/figures/fig_pareto_efficiency.png)

### 표

| Num Actions | Dense Head | GVLA Head | Speedup |
| ---: | ---: | ---: | ---: |
| `1,024` | `2.77 ms` | `0.136 ms` | `20.38x` |
| `32,768` | `13.05 ms` | `0.148 ms` | `88.03x` |
| `1,048,576` | `342.00 ms` | `0.142 ms` | `2410.31x` |

### 결과 해석

- dense는 `N`이 커질수록 빠르게 무거워짐
- GVLA는 거의 일정
- 그래서 large action space에서 차이가 더 크게 벌어짐

### 발표 멘트

여기서 중요한 건 숫자 하나보다 추세입니다.  
action space가 커질수록 GVLA 쪽이 더 유리해지는 모양이 꽤 분명하게 보입니다.

---

## Slide 17. Experiment 2 Background

### 제목

`Experiment 2. Cross-Backbone Transplant`

### 소제목

`한 모델 전용 trick인지 아닌지 보기`

### 슬라이드에 넣을 말

#### 배경

- local trick이면 임팩트가 약함

#### 세팅

- Octo-Base
- OpenVLA-7B
- RT-2-X
- pi0.5

### 발표 멘트

이 실험은 GVLA가 특정 backbone에서만 통하는 아이디어인지 아닌지를 보기 위한 실험입니다.

---

## Slide 18. Experiment 2 Result

### 제목

`Experiment 2. Universal Head Transplant`

### 소제목

`large action regime에서 반복적으로 비슷한 방향성이 나옴`

### 표

| Model | Actions | Dense ms | GVLA ms | Speedup | Dense MB | GVLA MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Octo-Base | `1,048,576` | `342.00` | `0.14` | `2410.31x` | `3072` | `0.059` |
| OpenVLA-7B | `1,048,576` | `5.21` | `0.12` | `44.02x` | `16384` | `0.313` |
| RT-2-X | `1,048,576` | `354.66` | `0.17` | `2072.11x` | `16384` | `0.313` |
| pi0.5 | `1,048,576` | `1.76` | `0.16` | `10.94x` | `2048` | `0.039` |

### 보조 표

| Model | `1k` Speedup | `32k` Speedup | `1M` Speedup |
| --- | ---: | ---: | ---: |
| Octo-Base | `20.38x` | `88.03x` | `2410.31x` |
| OpenVLA-7B | `0.34x` | `1.56x` | `44.02x` |
| RT-2-X | `31.31x` | `89.60x` | `2072.11x` |
| pi0.5 | `0.26x` | `0.50x` | `10.94x` |

### 결과 해석

- backbone마다 절대값은 다름
- 그래도 large `N`으로 갈수록 advantage가 커지는 패턴은 반복
- memory reduction도 꽤 큼

### 발표 멘트

모든 setting에서 무조건 이긴다고 말할 필요는 없습니다.  
오히려 큰 action space로 갈수록 강점이 커진다고 말하는 편이 훨씬 현실적이고 설득력 있습니다.

---

## Slide 19. Experiment 3 Background

### 제목

`Experiment 3. Orthogonality Ablation`

### 소제목

`직교성이 핵심이면, 깨졌을 때 성능도 같이 무너져야 함`

### 슬라이드에 넣을 말

#### 배경

- orthogonality가 진짜 핵심인지 확인 필요

#### 세팅

- orthogonal reg. 유무 비교
- collision rate
- unique code ratio
- row correlation

### 발표 멘트

이 실험은 orthogonality가 그냥 보기 좋아서 들어간 게 아니라는 점을 보여주기 위한 겁니다.

---

## Slide 20. Experiment 3 Result

### 제목

`Experiment 3. Orthogonality Result`

### 소제목

`직교성이 깨지면 code quality가 눈에 띄게 나빠짐`

### figure

![Orthogonality Ablation Figure](../experiments/results/figures/ablation_orthogonality_paper.png)

### 표

| Code Bits | Method | Collision Rate | Unique Code Ratio | Mean Abs Row Cosine |
| ---: | --- | ---: | ---: | ---: |
| `20` | GVLA-Net (Ours) | `0.6314` | `0.6325` | `0.0000` |
| `20` | GVLA w/o Orthogonal Reg. | `0.8946` | `0.2202` | `0.2084` |
| `22` | GVLA-Net (Ours) | `0.2205` | `0.8852` | `0.0000` |
| `22` | GVLA w/o Orthogonal Reg. | `0.7738` | `0.3752` | `0.1898` |
| `24` | GVLA-Net (Ours) | `0.0604` | `0.9695` | `0.0000` |
| `24` | GVLA w/o Orthogonal Reg. | `0.6303` | `0.5173` | `0.2113` |

### 결과 해석

- orthogonal reg.가 있으면 collision이 훨씬 낮음
- unique code utilization도 더 좋음
- 결국 중요한 건 bit 수가 아니라 bit 품질

### 발표 멘트

이 결과는 GVLA의 포인트가 그냥 binary code 자체가 아니라,  
좋은 binary question을 어떻게 만들 것인가에 있다는 걸 보여줍니다.

---

## Slide 21. Experiment 4 Background

### 제목

`Experiment 4. Mechanistic Analysis`

### 소제목

`정말 각 bit가 다른 정보를 주고 있나`

### 슬라이드에 넣을 말

#### 배경

- 속도만 보면 내부에서 왜 잘 되는지 설명이 약함

#### 세팅

- orthogonal vs random basis
- unique code rate
- info overlap
- entropy waterfall

### 발표 멘트

이 실험은 내부 메커니즘 설명용입니다.  
왜 잘 되는지에 대한 그림을 같이 보여주고 싶었습니다.

---

## Slide 22. Experiment 4A Result

### 제목

`Experiment 4A. Correlation / Overlap`

### 소제목

`orthogonal basis가 더 덜 겹침`

### figure

![Orthogonality Heatmap](../experiments/results/figures/fig1_orthogonality_heatmap_paper.png)

![Correlation Sweep](../experiments/results/figures/fig4_correlation_sweep_paper.png)

### 표

| N | Method | Unique Code Rate | Noise Accuracy | Ortho Error | Info Overlap |
| ---: | --- | ---: | ---: | ---: | ---: |
| `64` | Orthogonal W | `0.6563` | `0.5781` | `4.44e-07` | `3.65e-08` |
| `64` | Random W | `0.5781` | `0.7031` | `1.1859` | `0.1744` |
| `1024` | Orthogonal W | `0.6357` | `0.5371` | `4.18e-07` | `2.05e-08` |
| `1024` | Random W | `0.5420` | `0.5322` | `1.6300` | `0.1374` |
| `16384` | Orthogonal W | `0.6319` | `0.4100` | `6.17e-07` | `2.09e-08` |
| `16384` | Random W | `0.4647` | `0.4064` | `2.3276` | `0.1387` |

### 결과 해석

- orthogonal 쪽이 unique code rate가 더 높음
- info overlap은 거의 0에 가까움
- bit가 서로 다른 역할을 한다는 해석과 잘 맞음

### 발표 멘트

여기서는 직교성이 실제로 중복을 줄여주고 있다는 걸 보여주고 싶었습니다.

---

## Slide 23. Experiment 4B Result

### 제목

`Experiment 4B. Entropy Waterfall`

### 소제목

`bit가 늘수록 후보군이 빠르게 줄어듦`

### figure

![Entropy Waterfall](../experiments/results/figures/fig_entropy_waterfall_measured.png)

### 표

| Bit | Candidate Count | Entropy (bits) | Peak Probability |
| ---: | ---: | ---: | ---: |
| `1` | `8,388,608` | `22.10` | `3.18e-07` |
| `4` | `1,048,576` | `20.00` | `9.76e-07` |
| `8` | `65,536` | `16.00` | `1.53e-05` |
| `12` | `4,096` | `12.00` | `2.44e-04` |
| `16` | `256` | `8.00` | `3.91e-03` |
| `20` | `16` | `4.00` | `6.25e-02` |
| `24` | `1` | `0.00` | `1.00` |

### 결과 해석

- bit 수가 늘수록 candidate set이 기하급수적으로 줄어듦
- entropy도 같이 내려감
- 마지막에는 ambiguity가 거의 사라지는 모양

### 발표 멘트

이 figure는 GVLA가 어떻게 후보를 줄여 가는지 가장 직관적으로 보여줍니다.

---

## Slide 24. Experiment 5 Background

### 제목

`Experiment 5. Tracking / Control`

### 소제목

`이게 실제 control 쪽에서도 의미가 있을까`

### 슬라이드에 넣을 말

#### 배경

- head benchmark만으로는 practical relevance가 약함

#### 세팅

- `131,072 / 524,288 / 1,048,576`
- GVLA controller vs dense controller
- latency / FPS / final error 비교

### 발표 멘트

속도만 빠르다고 끝은 아니니까, control 관점에서도 한 번 보자는 실험입니다.

---

## Slide 25. Experiment 5 Result

### 제목

`Experiment 5. Tracking Result`

### 소제목

`action space가 커질수록 control 쪽 차이도 보이기 시작`

### figure

![Tracking Scaling](../experiments/results/figures/fig_tracking_scaling.png)

### 표

| Action Space | Controller | Mean Latency (ms) | FPS | Final Error | Arrival Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| `131,072` | GVLA | `1.492` | `670.36` | `0.0278` | `1.20` |
| `131,072` | Dense | `0.893` | `1119.22` | `0.0377` | `1.14` |
| `524,288` | GVLA | `1.672` | `598.00` | `0.0417` | `1.10` |
| `524,288` | Dense | `3.548` | `281.88` | `0.0376` | `1.14` |
| `1,048,576` | GVLA | `1.632` | `612.61` | `0.0277` | `1.20` |
| `1,048,576` | Dense | `6.847` | `146.06` | `0.0376` | `1.14` |

### 결과 해석

- `131k`에서는 dense가 더 빠르지만 GVLA error가 더 낮음
- `524k`, `1M`에서는 GVLA가 latency / FPS에서도 우세
- 큰 action space에서 practical benefit이 보일 가능성

### 발표 멘트

이 결과는 GVLA가 단순 synthetic head benchmark만 좋은 건 아닐 수 있다는 점을 보여줍니다.  
특히 action space가 커질수록 control 쪽에서도 이점이 보일 여지가 있습니다.

---

## Slide 26. Integrated Reading

### 제목

`지금까지 결과를 한 문장으로 묶으면`

### 소제목

`large action regime에서 action inference 구조를 다시 볼 필요가 있음`

### 슬라이드에 넣을 말

1. dense action scoring은 큰 action space에서 무거워짐
2. GVLA는 질문 기반 routing으로 구조를 바꿔보려 함
3. scaling 결과는 이 차이가 커질 수 있음을 보여줌
4. ablation은 orthogonality가 핵심임을 보여줌
5. mechanism 실험은 왜 그런지 설명해줌
6. tracking / transplant는 응용성과 범용성을 뒷받침

### 발표 멘트

그래서 지금 단계에서 제가 말하고 싶은 건 아주 단순합니다.  
large action space에서는 action inference 구조 자체를 다시 볼 필요가 있고, GVLA는 그 한 가지 꽤 강한 후보가 될 수 있다는 점입니다.

---

## Slide 27. Contributions

### 제목

`이 연구에서 새롭게 제안하는 것`

### 소제목

`method / structure / evidence`

### 슬라이드에 넣을 말

1. dense `O(N)` action scoring 대신 orthogonal `O(log N)` routing 관점 제안
2. learnable orthogonal projection layer 제안
3. scaling / ablation / entropy / tracking / transplant 실험으로 다각도 검증
4. speedup 숫자만이 아니라 내부 메커니즘도 같이 설명

### 발표 멘트

제 생각에 이 연구의 장점은 결과 하나가 아니라, 아이디어와 수식, 그리고 실험 해석이 같이 묶여 있다는 점입니다.

---

## Slide 28. Paper Outline

### 제목

`논문으로 쓰면 이런 구조`

### 소제목

`메시지도 비교적 깔끔하게 정리 가능`

### 슬라이드에 넣을 말

1. Introduction  
2. Related Work  
3. Method  
4. Why `log2(N)` / Why Orthogonality  
5. Experiments  
6. Discussion / Limitation  
7. Conclusion

### 발표 멘트

논문 구조도 비교적 자연스럽습니다.  
문제 제기, 이론적 동기, 방법, 실험, 해석으로 무리 없이 이어집니다.

---

## Slide 29. Final Message

### 제목

`마지막으로 하고 싶은 말`

### 소제목

`이건 단순한 head speedup 얘기만은 아님`

### 슬라이드에 넣을 말

- action inference를 다른 방식으로 볼 수 있는가
- large action regime에서 routing 구조를 바꿀 가치가 있는가
- 현재 결과들은 그 가능성을 꽤 강하게 보여주는 편

### 발표 멘트

제 생각에 GVLA의 포인트는 단순히 더 빠른 head 하나를 만드는 게 아닙니다.  
action inference를 dense comparison 말고 다른 방식으로 짤 수 있는지, 그리고 그게 실제로 의미가 있는지를 보여보자는 데 있습니다.

---

## PPT에 바로 쓸 figure 모음

### Scaling

![Pareto Efficiency Figure](../experiments/results/figures/fig_pareto_efficiency.png)

### Orthogonality Ablation

![Orthogonality Ablation Figure](../experiments/results/figures/ablation_orthogonality_paper.png)

### Orthogonality Heatmap

![Orthogonality Heatmap](../experiments/results/figures/fig1_orthogonality_heatmap_paper.png)

### Correlation Sweep

![Correlation Sweep](../experiments/results/figures/fig4_correlation_sweep_paper.png)

### Entropy Waterfall

![Entropy Waterfall](../experiments/results/figures/fig_entropy_waterfall_measured.png)

### Tracking Scaling

![Tracking Scaling](../experiments/results/figures/fig_tracking_scaling.png)

# Universal VLA Comparison Summary

이 디렉터리는 `experiments/universal_vla_comparison.py` 계열 실행 결과를 모아둔 곳이다. 폴더가 여러 개라 바로 보기 어렵기 때문에, 아래에 실행별 의미와 최신 기준 해석을 정리한다.

## Which Run To Trust First

가장 먼저 볼 기준 결과는 `20260426_110851_env_probe_smoke` 이다.

- `octo_env`, `openpi_env` 를 분리해서 probe한 실행이다.
- `args.txt` 에 `octo_python`, `openpi_python`, `third_party_root` 가 명시되어 있다.
- `Octo-Base` hidden width를 `768`, `pi0.5` width를 `1024`로 실제 로컬 환경에서 해석했다.

반대로 아래 실행들은 참고용이다.

- `20260426_004710_smoke_test`
  CPU 스모크 실행이다. `Octo-Base=1024`, `pi0=4096` 같은 proxy 값이 포함돼 있다.
- `20260426_004900_universal_vla_comparison`
  CUDA/FP16 실행이지만 절대 지연 시간이 지나치게 낮아 보인다. 방향성 확인용으로만 보는 편이 안전하다.
- `20260426_005709_pi05_smoke`
  `pi0.5`를 포함하지만 일부 값이 비정상적이다. 예를 들어 `Octo-Base @ 1,048,576 actions` 는 dense latency가 `0.0`으로 기록돼 있다.
- `20260426_005619_pi05_smoke`, `20260426_005649_pi05_smoke`
  `args.txt`만 있고 CSV 결과가 없다.

## Latest Run Snapshot

기준 파일:

- `20260426_110851_env_probe_smoke/universal_vla_comparison.csv`

핵심 컬럼 해석:

- `dense_head_latency_ms`: 기존 dense action head 지연 시간
- `gvla_head_latency_ms`: GVLA head 지연 시간
- `speedup_x`: dense 대비 GVLA 속도 개선 배수
- `memory_reduction_x`: dense 대비 GVLA 메모리 절감 배수

| Model | Actions | Dense ms | GVLA ms | Speedup | Dense MB | GVLA MB | Memory Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Octo-Base | 1,024 | 2.772 | 0.136 | 20.38x | 3.000 | 0.029 | 102.4x |
| Octo-Base | 32,768 | 13.052 | 0.148 | 88.03x | 96.000 | 0.044 | 2184.5x |
| Octo-Base | 1,048,576 | 342.002 | 0.142 | 2410.31x | 3072.000 | 0.059 | 52428.8x |
| OpenVLA-7B | 1,024 | 8.375 | 0.288 | 29.13x | 16.000 | 0.156 | 102.4x |
| OpenVLA-7B | 32,768 | 8.578 | 0.174 | 49.40x | 512.000 | 0.234 | 2184.5x |
| OpenVLA-7B | 1,048,576 | 15.091 | 0.169 | 89.11x | 16384.000 | 0.312 | 52428.8x |
| RT-2-X | 1,024 | 4.855 | 0.155 | 31.31x | 16.000 | 0.156 | 102.4x |
| RT-2-X | 32,768 | 15.455 | 0.172 | 89.60x | 512.000 | 0.234 | 2184.5x |
| RT-2-X | 1,048,576 | 354.657 | 0.171 | 2072.11x | 16384.000 | 0.312 | 52428.8x |
| pi0.5 | 1,024 | 0.188 | 0.255 | 0.74x | 4.000 | 0.039 | 102.4x |
| pi0.5 | 32,768 | 7.628 | 0.147 | 51.79x | 128.000 | 0.059 | 2184.5x |
| pi0.5 | 1,048,576 | 245.682 | 0.142 | 1734.33x | 4096.000 | 0.078 | 52428.8x |

## Fast Takeaways

1. GVLA는 action space가 커질수록 이득이 급격히 커진다.
2. `1,024 actions` 에서는 `pi0.5`가 오히려 GVLA보다 dense가 더 빠르다 (`0.74x`).
3. `32,768 actions` 부터는 모든 모델에서 GVLA가 우세하다.
4. `1,048,576 actions` 에서는 모든 모델이 매우 큰 속도 개선을 보인다. 가장 큰 개선은 최신 실행 기준 `Octo-Base`의 `2410.31x`다.
5. 메모리 절감 배수는 모델 종류보다 `num_actions`에 거의 의해 결정된다. 모든 모델에서
   `102.4x -> 2184.5x -> 52428.8x` 패턴을 보인다.

## Model-By-Model Read

- `Octo-Base`
  GVLA latency가 `0.136~0.148 ms` 수준으로 거의 일정하다. action 수가 커질수록 dense head가 급격히 불리해진다.
- `OpenVLA-7B`
  최신 실행에서는 세 구간 모두 GVLA 우세다. 특히 `1,048,576 actions`에서 `89.11x` 개선이 나온다.
- `RT-2-X`
  `1,024`와 `32,768 actions` 구간의 최고 speedup은 최신 실행 기준 RT-2-X다.
- `pi0.5`
  작은 action space에서는 dense가 이미 충분히 작아서 GVLA 이점이 없다. 하지만 큰 action space로 가면 개선 폭이 매우 커진다.

## Cross-Run Stability Notes

실행마다 절대 지연 시간은 꽤 흔들린다. 따라서 이 디렉터리에서는 절대 시간보다 아래 두 가지를 우선적으로 읽는 게 안전하다.

- action 수가 커질수록 GVLA 이점이 커진다는 방향성
- 메모리 절감 배수가 매우 크게 유지된다는 점

실행별 최고 speedup만 요약하면:

- `20260426_004710_smoke_test`
  `RT-2-X`가 각 action 구간에서 최고 speedup (`1.67x`, `40.13x`, `1251.16x`)
- `20260426_004900_universal_vla_comparison`
  소규모 action에서는 개선이 작고, 일부 항목은 `1x` 부근이다.
- `20260426_005709_pi05_smoke`
  수치 변동이 크고 일부 값이 비정상적이어서 참고용으로만 사용
- `20260426_110851_env_probe_smoke`
  현재 기준으로 가장 해석 가능한 결과

## Practical Reading Guide

논문/슬라이드/메모에서 빠르게 인용하려면 다음 문장으로 요약할 수 있다.

> 최신 `env_probe_smoke` 기준으로 GVLA는 large action regime에서 dense action head 대비 수십 배에서 수천 배까지 latency를 줄이고, 메모리는 최대 약 `52k`배까지 절감한다. 다만 작은 action regime, 특히 `pi0.5 @ 1,024 actions` 에서는 dense baseline이 더 빠를 수 있다.

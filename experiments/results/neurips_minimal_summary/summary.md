# NeurIPS Minimal CPU Summary

## Precision custom2.5 (200 rollouts)

| Setting | Successes | Rate | 95% CI |
|---|---:|---:|---:|
| continuous | 107/200 | 53.5% | [46.6, 60.3] |
| 256 | 24/200 | 12.0% | [8.2, 17.2] |
| 1024 | 86/200 | 43.0% | [36.3, 49.9] |
| 2048 | 94/200 | 47.0% | [40.2, 53.9] |

## Gray code main ablation

| M | Variant | Successes | Rate | 95% CI |
|---|---|---:|---:|---:|
| 128 | dense | 5/50 | 10.0% | [4.3, 21.4] |
| 128 | gvla | 1/50 | 2.0% | [0.4, 10.5] |
| 128 | gvla_gray | 8/50 | 16.0% | [8.3, 28.5] |
| 1024 | dense | 2/50 | 4.0% | [1.1, 13.5] |
| 1024 | gvla | 1/50 | 2.0% | [0.4, 10.5] |
| 1024 | gvla_gray | 9/50 | 18.0% | [9.8, 30.8] |

## Latency anchors

- Head-only N=1024: dense 0.2292 ms, GVLA 0.0396 ms, speedup 5.8x.
- Head-only N=4,194,304: dense 73.9956 ms, GVLA 0.0274 ms, speedup 2700.5x.
- Batch=1, M=65536: dense 0.6775 ms, GVLA 2.4728 ms.
- Batch=1024, M=65536: dense 14.3673 ms, GVLA 2.5114 ms.

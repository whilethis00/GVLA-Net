# Precision custom2.5 (200 rollouts)

Task: `pick_place_can_precision`

Config:
- `xy_tol=0.015`
- `release_clearance=0.075`
- `transport_xy_thresh=0.01`
- `place_height=0.068`
- `kp_place=5.4`

| Setting | Successes | Rate | 95% CI |
|---|---:|---:|---:|
| continuous | 107/200 | 53.5% | [46.6, 60.3] |
| 256 | 24/200 | 12.0% | [8.2, 17.2] |
| 1024 | 86/200 | 43.0% | [36.3, 49.9] |
| 2048 | 94/200 | 47.0% | [40.2, 53.9] |

Key message:
- `256` is clearly too low.
- `1024` and `2048` recover strongly.
- The relationship to `continuous` is visible in the same table.

# Table 3. Reviewer-Defense Controls at M=1024

Validation metrics used to answer the main reviewer attacks: single seed, orthogonality dependence, and arbitrary-code alternatives.

## Sources

- `results/final_submission_results.md`
- `experiments/results/bc_study/reviewer_defense_metrics/validation_metrics.json`

## Table

| Run | Encoding | L1 ↓ | Bin Err ↓ | Hamming ↓ | Exact ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| GVLA | Natural | 0.0554 | 28.2745 | 0.1938 | 0.2393 |
| GVLA seed2 | Natural | 0.0530 | 27.0196 | 0.1890 | 0.2408 |
| GVLA | Gray | 0.0304 | 15.4692 | 0.1498 | 0.2946 |
| GVLA no-orth | Gray | 0.0294 | 14.9637 | 0.1497 | 0.2934 |
| GVLA random | Random | 0.4020 | 205.7474 | 0.2840 | 0.1819 |

## Notes

- Natural seed2 stays close to Natural, reducing the single-seed artifact concern.
- Gray no-orth stays close to Gray, so the main effect is not explained by orthogonality regularization.
- Random degrades sharply, supporting the code-geometry interpretation.

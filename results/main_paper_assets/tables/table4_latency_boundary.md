# Table 4. Latency / Efficiency Boundary

Small main-paper summary that shows the paper is not hiding unfavorable latency evidence.

## Sources

- `results/final_submission_results.md`
- `experiments/results/bc_study/end_to_end_latency_gpu/end_to_end_latency.md`

## Table

| Measurement | Dense | Natural bitwise | Gray bitwise | Main conclusion |
| --- | ---: | ---: | ---: | --- |
| Matched GPU latency, M=2048, batch 1 | 0.7361 ms | 2.2069 ms | 5.9004 ms | Dense remains faster in the current BC setup |
| Matched GPU latency, M=2048, batch 256 | 0.7909 ms | 2.2364 ms | 5.6997 ms | No universal speedup claim |
| Bitwise-family latency | — | faster | slower | Gray is for learnability, not speed |
| Head-only scaling | avoids linear M-way output growth | — | — | efficiency potential only |

## Notes

- The main paper should describe latency as a qualified secondary analysis, not as primary evidence.
- Matched end-to-end latency up to `M<=2048` does not outperform Dense in the locked BC artifact.

# Dense nested sharing-gate boundary

Device: `cuda`; dense-boundary pass: **True**.

| Log drift | Refused | Refusal fraction (95% Wilson CI) | Shared RMSE | BIC support |
|---:|---:|---:|---:|---:|
| 0.05000 | 1/12 | 0.08 [0.01, 0.35] | 0.002382 | -904 |
| 0.05625 | 1/12 | 0.08 [0.01, 0.35] | 0.002464 | -478 |
| 0.06250 | 2/12 | 0.17 [0.05, 0.45] | 0.002656 | -315 |
| 0.06875 | 3/12 | 0.25 [0.09, 0.53] | 0.002814 | -449 |
| 0.07500 | 9/12 | 0.75 [0.47, 0.91] | 0.003155 | -671 |

Empirical 50% crossing bracket: [0.06875, 0.07500]; piecewise-linear estimate: 0.07187.

The inherited gate and thresholds were not modified for this experiment.

# Nested grouped-sharing gate

Device: `cuda`; route pass: **True**.

| Log drift | Noise corr. | Refused | Scope-limited | Group BIC support | Shared RMSE | Val. ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 | 0.00 | -758 | 0.0012 | 0.22 |
| 0.00 | 0.60 | 0.00 | 0.00 | -266 | 0.00106 | 0.373 |
| 0.05 | 0.00 | 0.00 | 0.00 | -495 | 0.00223 | 0.62 |
| 0.05 | 0.60 | 0.00 | 0.00 | -901 | 0.00231 | 0.347 |
| 0.15 | 0.00 | 1.00 | 0.00 | 113 | 0.00597 | 1.21 |
| 0.15 | 0.60 | 1.00 | 0.00 | 640 | 0.00605 | 3.53 |

BIC detects heterogeneity; refusal additionally requires material held-out degradation.

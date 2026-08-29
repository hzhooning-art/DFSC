# Decomposed model-and-noise tolerance sharing gate

Device: `cuda`; route pass: **True**.

Model allowance: `0.00133573`.

| Drift | True rho (diagnostic) | Noise-only refused | Decomposed refused | Proxy | Total limit | Shared RMSE | In scope |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.00 | 0.00 | 0.00 | 0.169 | 0.003571 | 0.001058 | 1.00 |
| 0.000 | 0.60 | 0.25 | 0.00 | 0.685 | 0.003636 | 0.001153 | 1.00 |
| 0.050 | 0.00 | 0.25 | 0.00 | 0.167 | 0.003571 | 0.002211 | 1.00 |
| 0.050 | 0.60 | 0.75 | 0.00 | 0.673 | 0.003635 | 0.002368 | 1.00 |
| 0.075 | 0.00 | 1.00 | 0.25 | 0.170 | 0.003571 | 0.002999 | 1.00 |
| 0.075 | 0.60 | 1.00 | 0.25 | 0.668 | 0.003634 | 0.003284 | 1.00 |
| 0.150 | 0.00 | 1.00 | 1.00 | 0.169 | 0.003571 | 0.005598 | 1.00 |
| 0.150 | 0.60 | 1.00 | 1.00 | 0.634 | 0.00363 | 0.005975 | 1.00 |

The model allowance is calibrated on an independent bank at the predeclared approximate-sharing boundary; project decisions use only the observed proxy.

# Observable correlation-aware sharing gate

Device: `cuda`; route pass: **False**.

| Drift | True rho (diagnostic) | Old refused | Noise-aware refused | Proxy | Calibrated limit | Shared RMSE | In scope |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.00 | 0.00 | 0.00 | 0.172 | 0.002077 | 0.001129 | 1.00 |
| 0.000 | 0.60 | 0.00 | 0.00 | 0.681 | 0.002681 | 0.00155 | 1.00 |
| 0.050 | 0.00 | 0.00 | 1.00 | 0.166 | 0.002069 | 0.002161 | 1.00 |
| 0.050 | 0.60 | 0.00 | 0.25 | 0.642 | 0.002635 | 0.002473 | 1.00 |
| 0.075 | 0.00 | 0.75 | 1.00 | 0.160 | 0.002062 | 0.003071 | 0.50 |
| 0.075 | 0.60 | 1.00 | 1.00 | 0.658 | 0.002654 | 0.003495 | 1.00 |
| 0.150 | 0.00 | 1.00 | 1.00 | 0.164 | 0.002068 | 0.00542 | 0.75 |
| 0.150 | 0.60 | 1.00 | 1.00 | 0.652 | 0.002646 | 0.006274 | 1.00 |

The true noise correlation is retained only as an evaluation diagnostic; the gate uses the observed second-difference proxy.

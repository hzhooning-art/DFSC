# Nested grouped-sharing refusal boundary

Device: `cuda`; boundary-map pass: **False**.

| Log drift | Noise corr. | Refused | Refusal fraction (95% Wilson CI) | BIC support | Shared RMSE | Val. ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.075 | 0.00 | 3/5 | 0.60 [0.23, 0.88] | -1.25e+03 | 0.00301 | 0.438 |
| 0.075 | 0.60 | 3/5 | 0.60 [0.23, 0.88] | -827 | 0.00327 | 0.481 |
| 0.100 | 0.00 | 5/5 | 1.00 [0.57, 1.00] | -877 | 0.00381 | 0.612 |
| 0.100 | 0.60 | 5/5 | 1.00 [0.57, 1.00] | 32.7 | 0.00423 | 1.43 |
| 0.125 | 0.00 | 5/5 | 1.00 [0.57, 1.00] | -312 | 0.00502 | 1.07 |
| 0.125 | 0.60 | 5/5 | 1.00 [0.57, 1.00] | -123 | 0.00495 | 1.29 |

The thresholds and decision rule are inherited unchanged from the preceding nested-gate experiment.

# Frozen multi-axis hierarchical contract audit

Route pass: **False**.

| Domain | Method | Coverage | Selective accuracy | False refusal | Severe refusal |
|:---|:---|---:|---:|---:|---:|
| Gaussian core | validation_axis | 0.896 | 0.953 | 0.062 | 1.000 |
| Gaussian core | structural_axis | 0.792 | 1.000 | 0.000 | 0.750 |
| Gaussian core | hierarchical | 0.750 | 1.000 | 0.000 | 0.750 |
| Gaussian stress | validation_axis | 1.000 | 0.750 | 0.375 | 1.000 |
| Gaussian stress | structural_axis | 0.792 | 0.947 | 0.000 | 0.375 |
| Gaussian stress | hierarchical | 0.000 | 0.000 | 0.000 | 0.000 |
| Real residual background | validation_axis | 1.000 | 0.806 | 0.292 | 1.000 |
| Real residual background | structural_axis | 0.750 | 0.963 | 0.042 | 0.583 |
| Real residual background | hierarchical | 0.639 | 0.957 | 0.042 | 0.583 |

The decision architecture and success criteria were frozen before this locked-output audit. No model was refit and no threshold was recalibrated.

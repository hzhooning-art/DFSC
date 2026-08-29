# Nonlinear misspecification transition boundary

- Route pass: **True**
- Repeats per strength: 6
- Wilson intervals are descriptive with six repeats per cell.
- Refusal uses the frozen numerical-quality gate plus the retrospective late audit.

| Strength | Refusals | Rate (95% Wilson) | Refusal causes: fit / condition / late | Rank-3 rate | Median condition | Median extrap. RMSE | Max extrap. RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0/6 | 0.000 [0.000, 0.390] | 0 / 0 / 0 | 1.000 | 3.981e+02 | 2.389e-03 | 3.594e-03 |
| 0.075 | 0/6 | 0.000 [0.000, 0.390] | 0 / 0 / 0 | 1.000 | 5.446e+02 | 3.667e-03 | 4.316e-03 |
| 0.100 | 6/6 | 1.000 [0.610, 1.000] | 6 / 0 / 0 | 1.000 | 8.221e+02 | 4.797e-03 | 5.686e-03 |
| 0.125 | 6/6 | 1.000 [0.610, 1.000] | 6 / 0 / 0 | 0.833 | 9.277e+02 | 5.550e-03 | 9.333e-03 |
| 0.150 | 6/6 | 1.000 [0.610, 1.000] | 6 / 0 / 0 | 0.333 | 1.094e+02 | 8.947e-03 | 9.718e-03 |
| 0.175 | 6/6 | 1.000 [0.610, 1.000] | 5 / 1 / 0 | 0.167 | 1.346e+02 | 8.667e-03 | 9.026e-03 |
| 0.200 | 6/6 | 1.000 [0.610, 1.000] | 2 / 4 / 0 | 0.000 | 2.253e+15 | 8.517e-03 | 8.969e-03 |

## Prespecified checks

- lowest_strength_has_at_least_one_accepted_fit: **True**
- highest_strength_has_at_least_one_refusal: **True**
- no_silent_late_audit_failures: **True**
- observed_refusal_rate_is_nondecreasing: **True**

The reported boundary is an observed transition under this fixed design,
not a population-calibrated refusal probability or an a-priori guarantee.

# Calibrated multi-window refusal probe

- Horizon: 18
- Noise std: 6.0e-04
- Diagnostic windows: early [15%,35%), middle [45%,65%), late [80%,100%)
- Calibrated lag-1 threshold: 0.205948
- Route pass: **True**

| Family | Level | Terminal refusal | Multi-window refusal | Change |
|---|---:|---:|---:|---:|
| signed_residue | zero | 0.000 | 0.000 | +0.000 |
| signed_residue | above | 0.750 | 1.000 | +0.250 |
| oscillation | zero | 0.000 | 0.000 | +0.000 |
| oscillation | above | 0.000 | 1.000 | +1.000 |

## Checks

- heldout_zero_false_refusal_at_most_0.25: **True**
- oscillation_detection_gain_at_least_0.50: **True**
- signed_detection_not_degraded: **True**
- accepted_elevated_rank_absorption_at_most_0.25: **True**

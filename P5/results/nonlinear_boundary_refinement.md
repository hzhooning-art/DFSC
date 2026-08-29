# Refined nonlinear-refusal boundary

- Route pass: **True**
- Frozen training-fit threshold: 3.200e-03 (4.00 noise standard deviations)
- Eight repeats per strength; Wilson intervals remain descriptive.

| Strength | Refusals | Rate (95% Wilson) | Rank-3 rate | Median train RMSE/noise | Range | Median clean-train RMSE | Median extrap. RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.075 | 1/8 | 0.125 [0.022, 0.471] | 1.000 | 3.559 | [3.077, 4.552] | 2.708e-03 | 3.876e-03 |
| 0.080 | 3/8 | 0.375 [0.137, 0.694] | 1.000 | 3.883 | [1.979, 5.542] | 3.015e-03 | 3.949e-03 |
| 0.085 | 7/8 | 0.875 [0.529, 0.978] | 1.000 | 4.400 | [3.403, 5.262] | 3.427e-03 | 4.979e-03 |
| 0.090 | 7/8 | 0.875 [0.529, 0.978] | 1.000 | 4.809 | [3.996, 6.011] | 3.808e-03 | 5.098e-03 |
| 0.095 | 8/8 | 1.000 [0.676, 1.000] | 1.000 | 4.726 | [4.383, 5.814] | 3.692e-03 | 4.997e-03 |
| 0.100 | 8/8 | 1.000 [0.676, 1.000] | 1.000 | 5.210 | [4.660, 6.372] | 4.058e-03 | 5.271e-03 |

## Prespecified checks

- both_acceptance_and_refusal_are_observed: **True**
- majority_transition_bracket_is_at_most_0p01: **True**
- observed_refusal_rate_is_nondecreasing: **True**
- no_silent_late_audit_failures: **True**

## Exploratory diagnostic

Across the six group medians, strength and train RMSE/noise had Spearman rho=0.943 (nominal p=0.004805). This is a descriptive six-group diagnostic, not a calibrated inferential result.

The boundary is a frozen protocol threshold under one noise/horizon design, not a physical phase transition.

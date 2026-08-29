# Single-series robust noise gate

- Route pass: **True**
- Noise scale is estimated from a dense single-series training prefix.
- The robust multiscale estimator is compared with a naive second-difference standard deviation.

| Noise kind | Base noise | Strength | Robust refusals (95% Wilson) | Median robust scale/base | Median naive scale/base | Median extrap./base |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 4.0e-04 | 0.050 | 1/2 [0.095, 0.905] | 1.204 | 2.457 | 5.553 |
| gaussian | 4.0e-04 | 0.085 | 2/2 [0.342, 1.000] | 1.077 | 2.321 | 11.874 |
| gaussian | 4.0e-04 | 0.200 | 2/2 [0.342, 1.000] | 1.121 | 1.949 | 21.931 |
| gaussian | 8.0e-04 | 0.050 | 0/2 [0.000, 0.658] | 1.127 | 1.508 | 2.709 |
| gaussian | 8.0e-04 | 0.085 | 0/2 [0.000, 0.658] | 1.052 | 1.436 | 5.947 |
| gaussian | 8.0e-04 | 0.200 | 2/2 [0.342, 1.000] | 1.133 | 1.317 | 10.966 |
| gaussian | 1.6e-03 | 0.050 | 0/2 [0.000, 0.658] | 1.063 | 1.155 | 1.241 |
| gaussian | 1.6e-03 | 0.085 | 0/2 [0.000, 0.658] | 0.990 | 1.108 | 3.098 |
| gaussian | 1.6e-03 | 0.200 | 2/2 [0.342, 1.000] | 1.083 | 1.103 | 5.485 |
| contaminated | 4.0e-04 | 0.050 | 0/2 [0.000, 0.658] | 1.289 | 2.956 | 5.262 |
| contaminated | 4.0e-04 | 0.085 | 2/2 [0.342, 1.000] | 1.108 | 2.843 | 13.878 |
| contaminated | 4.0e-04 | 0.200 | 2/2 [0.342, 1.000] | 1.288 | 2.497 | 19.531 |
| contaminated | 8.0e-04 | 0.050 | 0/2 [0.000, 0.658] | 1.231 | 2.212 | 2.612 |
| contaminated | 8.0e-04 | 0.085 | 1/2 [0.095, 0.905] | 1.128 | 2.174 | 7.108 |
| contaminated | 8.0e-04 | 0.200 | 2/2 [0.342, 1.000] | 1.182 | 2.043 | 9.788 |
| contaminated | 1.6e-03 | 0.050 | 0/2 [0.000, 0.658] | 1.216 | 1.976 | 1.529 |
| contaminated | 1.6e-03 | 0.085 | 0/2 [0.000, 0.658] | 1.140 | 1.970 | 3.558 |
| contaminated | 1.6e-03 | 0.200 | 2/2 [0.342, 1.000] | 1.174 | 1.912 | 4.917 |

## Gate diagnostics

- robust_normalized_gate: {'accepted': 18, 'refused': 18, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 7, 'max_accepted_extrapolation_rmse_over_noise': 8.727913655422403, 'median_accepted_extrapolation_rmse_over_noise': 3.3531662834186893}
- naive_normalized_gate: {'accepted': 24, 'refused': 12, 'silent_relative_extrapolation_failures': 4, 'refusals_without_relative_extrapolation_failure': 5, 'max_accepted_extrapolation_rmse_over_noise': 14.175204388616763, 'median_accepted_extrapolation_rmse_over_noise': 3.7873476454124266}
- oracle_normalized_gate: {'accepted': 14, 'refused': 22, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 11, 'max_accepted_extrapolation_rmse_over_noise': 4.314214017307047, 'median_accepted_extrapolation_rmse_over_noise': 2.6455126405207716}
- fixed_absolute_gate: {'accepted': 12, 'refused': 24, 'silent_relative_extrapolation_failures': 1, 'refusals_without_relative_extrapolation_failure': 14, 'max_accepted_extrapolation_rmse_over_noise': 11.12580833193595, 'median_accepted_extrapolation_rmse_over_noise': 2.6668390037339345}

## Prespecified feasibility checks

- median_robust_scale_error_within_limit: **True**
- p90_robust_scale_error_within_limit: **True**
- robust_gate_agrees_with_oracle_at_prespecified_rate: **True**
- robust_gate_has_no_silent_relative_extrapolation_failures: **True**
- robust_gate_controls_accepted_relative_extrapolation_error: **True**
- robust_estimator_less_inflated_than_naive_under_contamination: **True**
- all_cells_have_expected_repeat_count: **True**

The estimator requires one densely sampled calibration prefix, iid core noise, smooth signal curvature, and sparse gross contamination. It is not validated for temporal correlation or an arbitrarily sparse series.

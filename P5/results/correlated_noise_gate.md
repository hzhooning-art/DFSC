# Correlation-aware noise gate

- Route pass: **True**
- Observation noise follows a stationary AR(1) process.
- Unidentifiable correlation forces refusal.

| rho | Strength | Identifiable | Aware refusals (95% Wilson) | Median estimated rho | Median aware scale/base | Median iid scale/base | Median extrap./base |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.050 | 2/2 | 0/2 [0.000, 0.658] | -0.013 | 1.062 | 1.107 | 2.596 |
| 0.00 | 0.085 | 2/2 | 0/2 [0.000, 0.658] | 0.130 | 1.122 | 1.038 | 6.180 |
| 0.00 | 0.200 | 0/2 | 2/2 [0.342, 1.000] | 0.330 | 1.259 | 0.985 | 10.047 |
| 0.30 | 0.050 | 2/2 | 0/2 [0.000, 0.658] | 0.242 | 0.965 | 0.807 | 1.921 |
| 0.30 | 0.085 | 2/2 | 0/2 [0.000, 0.658] | 0.337 | 1.050 | 0.807 | 6.495 |
| 0.30 | 0.200 | 1/2 | 2/2 [0.342, 1.000] | 0.480 | 1.210 | 0.752 | 10.281 |
| 0.60 | 0.050 | 2/2 | 0/2 [0.000, 0.658] | 0.448 | 0.907 | 0.698 | 2.403 |
| 0.60 | 0.085 | 2/2 | 1/2 [0.095, 0.905] | 0.518 | 0.956 | 0.513 | 5.509 |
| 0.60 | 0.200 | 2/2 | 2/2 [0.342, 1.000] | 0.617 | 1.045 | 0.566 | 11.245 |
| 0.85 | 0.050 | 2/2 | 0/2 [0.000, 0.658] | 0.630 | 0.656 | 0.328 | 3.863 |
| 0.85 | 0.085 | 1/2 | 2/2 [0.342, 1.000] | 0.685 | 0.731 | 0.374 | 5.512 |
| 0.85 | 0.200 | 2/2 | 2/2 [0.342, 1.000] | 0.728 | 0.792 | 0.375 | 10.339 |

## Gate diagnostics

- aware_normalized_gate: {'accepted': 13, 'refused': 11, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 5, 'max_accepted_extrapolation_rmse_over_noise': 7.378830119250226, 'median_accepted_extrapolation_rmse_over_noise': 3.620670427743709}
- iid_normalized_gate: {'accepted': 7, 'refused': 17, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 11, 'max_accepted_extrapolation_rmse_over_noise': 5.57301006229692, 'median_accepted_extrapolation_rmse_over_noise': 3.1570255739075517}
- oracle_normalized_gate: {'accepted': 12, 'refused': 12, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 6, 'max_accepted_extrapolation_rmse_over_noise': 5.610298494190215, 'median_accepted_extrapolation_rmse_over_noise': 3.4342827985828133}

## Prespecified feasibility checks

- at_least_half_of_cases_are_identifiable: **True**
- median_aware_scale_error_within_limit: **True**
- p90_aware_scale_error_within_limit: **True**
- aware_gate_agrees_with_oracle_at_prespecified_rate: **True**
- aware_gate_has_no_silent_relative_extrapolation_failures: **True**
- aware_gate_controls_accepted_relative_extrapolation_error: **True**
- all_unidentifiable_cases_are_refused: **True**
- aware_gate_not_less_safe_than_iid_gate: **True**
- all_cells_have_expected_repeat_count: **True**

The correlation model is stationary AR(1), shared only at the model-class level across channels, and estimated from one dense calibration prefix. Long-memory, nonstationary, and irregularly sampled noise remain untested.

# Replicate-estimated gate under heteroscedastic noise

- Route pass: **True**
- Two independent replicates are averaged for fitting.
- The effective training noise is estimated from their paired difference.
- Noise amplitude increases with time and is normalized to the declared base RMS.

| Base noise | Strength | Estimated refusals (95% Wilson) | Median noise-scale error | Median train/estimated noise | Median extrap./oracle noise |
|---:|---:|---:|---:|---:|---:|
| 4.0e-04 | 0.050 | 3/3 [0.439, 1.000] | 0.024 | 8.922 | 7.438 |
| 4.0e-04 | 0.085 | 3/3 [0.439, 1.000] | 0.022 | 15.441 | 13.732 |
| 4.0e-04 | 0.200 | 3/3 [0.439, 1.000] | 0.033 | 43.590 | 23.934 |
| 8.0e-04 | 0.050 | 3/3 [0.439, 1.000] | 0.024 | 4.542 | 3.774 |
| 8.0e-04 | 0.085 | 3/3 [0.439, 1.000] | 0.022 | 7.788 | 6.658 |
| 8.0e-04 | 0.200 | 3/3 [0.439, 1.000] | 0.033 | 21.834 | 11.966 |
| 1.6e-03 | 0.050 | 0/3 [0.000, 0.561] | 0.024 | 2.405 | 1.919 |
| 1.6e-03 | 0.085 | 2/3 [0.208, 0.939] | 0.022 | 4.027 | 3.259 |
| 1.6e-03 | 0.200 | 3/3 [0.439, 1.000] | 0.033 | 10.974 | 5.983 |

## Gate-level diagnostics

- estimated_normalized_gate: {'accepted': 4, 'refused': 23, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 14, 'max_accepted_extrapolation_rmse_over_noise': 3.258679519290121, 'median_accepted_extrapolation_rmse_over_noise': 2.0323371362474028}
- oracle_normalized_gate: {'accepted': 5, 'refused': 22, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 13, 'max_accepted_extrapolation_rmse_over_noise': 3.258679519290121, 'median_accepted_extrapolation_rmse_over_noise': 2.145339552180324}
- fixed_absolute_gate: {'accepted': 11, 'refused': 16, 'silent_relative_extrapolation_failures': 1, 'refusals_without_relative_extrapolation_failure': 8, 'max_accepted_extrapolation_rmse_over_noise': 10.583073595030056, 'median_accepted_extrapolation_rmse_over_noise': 4.393450144163317}

## Prespecified checks

- replicate_noise_estimator_within_relative_error_limit: **True**
- estimated_gate_agrees_with_oracle_at_prespecified_rate: **True**
- estimated_gate_has_no_silent_relative_extrapolation_failures: **True**
- estimated_gate_controls_accepted_relative_extrapolation_error: **True**
- all_cells_have_expected_repeat_count: **True**

Repeated measurements identify the aggregate training noise scale in this controlled Gaussian experiment. This is not a guarantee for single-series, correlated, or non-Gaussian observations.

# Noise-normalized gate transfer

- Route pass: **True**
- Normalized training gate: RMSE/noise <= 4.0
- Relative extrapolation audit target: RMSE/noise <= 10.0
- Noise levels are paired through the same standard-normal draw and sampling indices.

| Noise | Strength | Fixed refusals | Normalized refusals (95% Wilson) | Median train/noise | Median extrap./noise |
|---:|---:|---:|---:|---:|---:|
| 4.0e-04 | 0.075 | 0/4 | 4/4 [0.510, 1.000] | 6.859 | 10.490 |
| 4.0e-04 | 0.085 | 4/4 | 4/4 [0.510, 1.000] | 8.668 | 11.912 |
| 4.0e-04 | 0.125 | 4/4 | 4/4 [0.510, 1.000] | 13.105 | 14.157 |
| 4.0e-04 | 0.200 | 4/4 | 4/4 [0.510, 1.000] | 22.718 | 21.005 |
| 8.0e-04 | 0.075 | 0/4 | 0/4 [0.000, 0.490] | 3.553 | 5.120 |
| 8.0e-04 | 0.085 | 4/4 | 4/4 [0.510, 1.000] | 4.379 | 6.038 |
| 8.0e-04 | 0.125 | 4/4 | 4/4 [0.510, 1.000] | 6.505 | 7.273 |
| 8.0e-04 | 0.200 | 4/4 | 4/4 [0.510, 1.000] | 11.364 | 10.518 |
| 1.6e-03 | 0.075 | 1/4 | 0/4 [0.000, 0.490] | 1.979 | 2.441 |
| 1.6e-03 | 0.085 | 4/4 | 0/4 [0.000, 0.490] | 2.331 | 3.113 |
| 1.6e-03 | 0.125 | 4/4 | 1/4 [0.046, 0.699] | 3.269 | 3.464 |
| 1.6e-03 | 0.200 | 4/4 | 4/4 [0.510, 1.000] | 5.719 | 5.278 |

## Gate-level diagnostics

- noise_normalized_gate: {'accepted': 15, 'refused': 33, 'silent_relative_extrapolation_failures': 0, 'refusals_without_relative_extrapolation_failure': 16, 'max_accepted_extrapolation_rmse_over_noise': 6.0164564212183755, 'median_accepted_extrapolation_rmse_over_noise': 3.1599854482808123}
- fixed_absolute_gate: {'accepted': 11, 'refused': 37, 'silent_relative_extrapolation_failures': 2, 'refusals_without_relative_extrapolation_failure': 22, 'max_accepted_extrapolation_rmse_over_noise': 12.109058771906511, 'median_accepted_extrapolation_rmse_over_noise': 5.78272296023396}

## Prespecified checks

- paired_gate_decisions_differ_on_at_least_one_case: **True**
- normalized_gate_has_no_silent_relative_extrapolation_failures: **True**
- normalized_gate_controls_accepted_relative_extrapolation_error: **True**
- all_cells_have_expected_repeat_count: **True**

The relative extrapolation limit is an operational audit target, not a theoretical error bound or a universal utility threshold.

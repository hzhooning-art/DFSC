# Unconstrained smoothing-spline baseline

- Route pass: **True**
- Smoothing was fixed from the known noise variance; no held-out values tuned it.
- Spline complexity and memory rank are not treated as equivalent quantities.

| Case | Memory extrap. | Spline train | Spline interp. | Spline extrap. |
|---|---:|---:|---:|---:|
| rank1_separated | 1.1843e-04 | 5.7075e-04 | 6.8326e-04 | 1.4390e+00 |
| rank2_separated | 1.5215e-04 | 5.4790e-04 | 7.4729e-04 | 5.8555e-01 |
| rank3_separated | 6.6199e-03 | 5.4849e-04 | 5.8197e-04 | 5.6488e-01 |

## Prespecified checks

- spline_training_rmse_reaches_2p5_noise_std_in_each_case: **True**
- spline_interpolation_rmse_within_5_noise_std_in_each_case: **True**
- mechanism_extrapolation_at_least_25_percent_better_than_spline_on_median: **True**
- memory_rank_recovered_in_at_least_7_of_9_trials: **True**

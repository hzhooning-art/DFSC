# Controlled mechanism misspecification

- Route pass: **True**
- Late-audit RMSE refusal limit: 1.000e-02
- The late audit requires observations beyond the training horizon; it is not an a-priori guarantee.

| Family | Level | Strength | Rank counts | Refusals | Memory extrap. | Modal extrap. | Spline extrap. | Ratio to best |
|---|---|---:|---|---:|---:|---:|---:|---:|
| rate_drift | control | 0.000 | {'1': 0, '2': 2, '3': 0} | 0/2 | 2.2429e-04 | 5.8299e-02 | 2.0578e+00 | 0.004 |
| rate_drift | mild | 0.250 | {'1': 0, '2': 2, '3': 0} | 0/2 | 2.8210e-03 | 4.9736e-02 | 4.1556e-01 | 0.057 |
| rate_drift | strong | 0.750 | {'1': 0, '2': 2, '3': 0} | 0/2 | 6.4238e-03 | 8.3356e-02 | 9.5884e-01 | 0.077 |
| nonlinear_feedback | control | 0.000 | {'1': 0, '2': 2, '3': 0} | 0/2 | 1.8706e-04 | 8.1798e-02 | 6.8576e-01 | 0.002 |
| nonlinear_feedback | mild | 0.050 | {'1': 0, '2': 0, '3': 2} | 0/2 | 2.2945e-03 | 2.3733e-02 | 8.2694e-01 | 0.097 |
| nonlinear_feedback | strong | 0.200 | {'1': 0, '2': 2, '3': 0} | 2/2 | 8.8906e-03 | 3.2362e-02 | 3.7716e-01 | 0.275 |

## Prespecified checks

- all_control_instances_have_sub_0p005_mechanism_extrapolation: **True**
- mild_misspecification_mechanism_not_over_25_percent_worse_than_best_trajectory: **True**
- every_strong_instance_is_useful_or_refused: **True**
- no_silent_late_audit_failures: **True**

A refusal means that an observed late-time audit exceeded the frozen error
limit or the numerical fit failed its quality gate. It does not predict
failure before late observations become available.

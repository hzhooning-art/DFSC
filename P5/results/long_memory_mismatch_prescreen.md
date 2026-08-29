# Long-memory mismatch prescreen

- Route pass: **False**
- Control AR(1) adequacy rate: 0.333
- Strong-memory detection rate: 0.583
- The expensive mechanism-fit matrix is skipped when this prescreen fails.

| d | Strength | Repeat | Estimated AR1 rho | Whiteness p | Adequate |
|---:|---:|---:|---:|---:|:---:|
| 0.00 | 0.050 | 0 | -0.010 | 0.141 | True |
| 0.00 | 0.050 | 1 | 0.060 | 0.00694 | False |
| 0.00 | 0.085 | 0 | 0.090 | 0.0133 | True |
| 0.00 | 0.085 | 1 | 0.160 | 1.05e-05 | False |
| 0.00 | 0.200 | 0 | 0.280 | 2.52e-07 | False |
| 0.00 | 0.200 | 1 | 0.410 | 0.00134 | False |
| 0.15 | 0.050 | 0 | 0.145 | 0.0139 | True |
| 0.15 | 0.050 | 1 | 0.080 | 0.000246 | False |
| 0.15 | 0.085 | 0 | 0.280 | 0.014 | True |
| 0.15 | 0.085 | 1 | 0.250 | 0.000121 | False |
| 0.15 | 0.200 | 0 | 0.380 | 0.0423 | True |
| 0.15 | 0.200 | 1 | 0.425 | 0.0184 | True |
| 0.30 | 0.050 | 0 | 0.180 | 0.00228 | False |
| 0.30 | 0.050 | 1 | 0.190 | 0.00058 | False |
| 0.30 | 0.085 | 0 | 0.285 | 4.15e-07 | False |
| 0.30 | 0.085 | 1 | 0.305 | 0.0784 | True |
| 0.30 | 0.200 | 0 | 0.480 | 0.00236 | False |
| 0.30 | 0.200 | 1 | 0.470 | 0.0233 | True |
| 0.45 | 0.050 | 0 | 0.335 | 0.0628 | True |
| 0.45 | 0.050 | 1 | 0.340 | 0.00658 | False |
| 0.45 | 0.085 | 0 | 0.460 | 0.0988 | True |
| 0.45 | 0.085 | 1 | 0.515 | 0.24 | True |
| 0.45 | 0.200 | 0 | 0.650 | 0.00554 | False |
| 0.45 | 0.200 | 1 | 0.665 | 3.17e-06 | False |

## Frozen checks

- all_prescreen_cells_complete: **True**
- control_ar1_adequacy_rate_within_limit: **False**
- strong_long_memory_detection_rate_within_limit: **False**

The pooled Ljung-Box diagnostic does not separate the iid control from strong finite-sample ARFIMA noise reliably enough under the frozen thresholds. The route is rejected without tuning the test after observing these outcomes.

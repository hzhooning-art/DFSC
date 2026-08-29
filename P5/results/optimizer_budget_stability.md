# Paired optimizer-budget decision-stability audit

Device: `cuda`; route pass: **False**.

Exact tri-state agreement: 0.500; direct retain/refuse reversal: 0.458; pairs containing an indeterminate result: 0.042.

| Channels | Noise std | Construction | Agreement | Direct reversal | Indeterminate |
|---:|---:|:---|---:|---:|---:|
| 32 | 4.0e-04 | antisymmetric | 0.00 | 1.00 | 0.00 |
| 32 | 4.0e-04 | curved | 0.00 | 1.00 | 0.00 |
| 32 | 1.6e-03 | antisymmetric | 1.00 | 0.00 | 0.00 |
| 32 | 1.6e-03 | curved | 1.00 | 0.00 | 0.00 |
| 128 | 4.0e-04 | antisymmetric | 0.33 | 0.67 | 0.00 |
| 128 | 4.0e-04 | curved | 0.00 | 1.00 | 0.00 |
| 128 | 1.6e-03 | antisymmetric | 1.00 | 0.00 | 0.00 |
| 128 | 1.6e-03 | curved | 0.67 | 0.00 | 0.33 |

All observations, initializations, calibrations, and scientific-decision thresholds are paired across budgets. No budget is selected after observing its result.

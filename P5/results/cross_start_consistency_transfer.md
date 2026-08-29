# Cross-start consistency transfer audit

Device: `cuda`; route pass: **False**.

| Channels | Noise std | Construction | Drift | Refused | Indeterminate | Adverse | Consistent starts | In scope |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 32 | 4.0e-04 | antisymmetric | 0.00 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 32 | 4.0e-04 | antisymmetric | 0.05 | 0.00 | 0.33 | 0.33 | 3.0 | 1.00 |
| 32 | 4.0e-04 | antisymmetric | 0.15 | 1.00 | 0.00 | 1.00 | 2.0 | 1.00 |
| 32 | 4.0e-04 | curved | 0.00 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 32 | 4.0e-04 | curved | 0.05 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 32 | 4.0e-04 | curved | 0.15 | 0.00 | 1.00 | 1.00 | 1.0 | 1.00 |
| 32 | 1.6e-03 | antisymmetric | 0.00 | 0.33 | 0.00 | 0.33 | 4.0 | 1.00 |
| 32 | 1.6e-03 | antisymmetric | 0.05 | 0.33 | 0.00 | 0.33 | 4.0 | 1.00 |
| 32 | 1.6e-03 | antisymmetric | 0.15 | 0.67 | 0.33 | 1.00 | 2.0 | 1.00 |
| 32 | 1.6e-03 | curved | 0.00 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 32 | 1.6e-03 | curved | 0.05 | 0.00 | 0.33 | 0.33 | 2.0 | 1.00 |
| 32 | 1.6e-03 | curved | 0.15 | 0.33 | 0.67 | 1.00 | 1.0 | 1.00 |
| 128 | 4.0e-04 | antisymmetric | 0.00 | 0.00 | 0.00 | 0.00 | 3.0 | 1.00 |
| 128 | 4.0e-04 | antisymmetric | 0.05 | 0.00 | 0.00 | 0.00 | 3.0 | 1.00 |
| 128 | 4.0e-04 | antisymmetric | 0.15 | 0.33 | 0.67 | 1.00 | 1.0 | 1.00 |
| 128 | 4.0e-04 | curved | 0.00 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 128 | 4.0e-04 | curved | 0.05 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 128 | 4.0e-04 | curved | 0.15 | 0.33 | 0.67 | 1.00 | 1.0 | 1.00 |
| 128 | 1.6e-03 | antisymmetric | 0.00 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 128 | 1.6e-03 | antisymmetric | 0.05 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 128 | 1.6e-03 | antisymmetric | 0.15 | 0.67 | 0.33 | 1.00 | 2.0 | 1.00 |
| 128 | 1.6e-03 | curved | 0.00 | 0.00 | 0.00 | 0.00 | 4.0 | 1.00 |
| 128 | 1.6e-03 | curved | 0.05 | 0.33 | 0.00 | 0.33 | 4.0 | 1.00 |
| 128 | 1.6e-03 | curved | 0.15 | 0.33 | 0.67 | 1.00 | 1.0 | 1.00 |

Calibration and project seeds are disjoint. Cross-start thresholds are the maximum exact-control second-start gaps and are never updated from project records.

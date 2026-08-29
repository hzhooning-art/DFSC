# Independent optimizer-budget consensus and abstention audit

Device: `cuda`; route pass: **False**.

Exact/mild false refusal: 0.188; exact/mild retained: 0.667; severe refused: 0.958; budget-sensitive: 0.097.

| Channels | Noise std | Construction | Drift | Retain | Refuse | Indeterminate | Budget-sensitive |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 32 | 4.0e-04 | antisymmetric | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 32 | 4.0e-04 | antisymmetric | 0.05 | 0.00 | 0.00 | 1.00 | 1.00 |
| 32 | 4.0e-04 | antisymmetric | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |
| 32 | 4.0e-04 | curved | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 32 | 4.0e-04 | curved | 0.05 | 0.67 | 0.00 | 0.33 | 0.33 |
| 32 | 4.0e-04 | curved | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |
| 32 | 1.6e-03 | antisymmetric | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 32 | 1.6e-03 | antisymmetric | 0.05 | 0.33 | 0.67 | 0.00 | 0.00 |
| 32 | 1.6e-03 | antisymmetric | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |
| 32 | 1.6e-03 | curved | 0.00 | 0.67 | 0.33 | 0.00 | 0.00 |
| 32 | 1.6e-03 | curved | 0.05 | 0.67 | 0.33 | 0.00 | 0.00 |
| 32 | 1.6e-03 | curved | 0.15 | 0.00 | 0.67 | 0.33 | 0.00 |
| 128 | 4.0e-04 | antisymmetric | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 128 | 4.0e-04 | antisymmetric | 0.05 | 0.67 | 0.00 | 0.33 | 0.33 |
| 128 | 4.0e-04 | antisymmetric | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |
| 128 | 4.0e-04 | curved | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 128 | 4.0e-04 | curved | 0.05 | 0.33 | 0.00 | 0.67 | 0.67 |
| 128 | 4.0e-04 | curved | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |
| 128 | 1.6e-03 | antisymmetric | 0.00 | 0.33 | 0.67 | 0.00 | 0.00 |
| 128 | 1.6e-03 | antisymmetric | 0.05 | 0.33 | 0.67 | 0.00 | 0.00 |
| 128 | 1.6e-03 | antisymmetric | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |
| 128 | 1.6e-03 | curved | 0.00 | 0.67 | 0.33 | 0.00 | 0.00 |
| 128 | 1.6e-03 | curved | 0.05 | 1.00 | 0.00 | 0.00 | 0.00 |
| 128 | 1.6e-03 | curved | 0.15 | 0.00 | 1.00 | 0.00 | 0.00 |

The rule, budgets, thresholds, and exit conditions were frozen before the independent project seeds were evaluated.

# Near-boundary contract-refusal probe

Labels define the post-hoc scoring axis only; the decision rule uses fit diagnostics.

| Family | Violation strength | Accept fraction | Median validation RMSE | Median |lag-1| | Selected ranks |
|---|---:|---:|---:|---:|---|
| oscillation | 0 | 1.00 | 0.000582 | 0.105 | [1, 1, 1] |
| oscillation | 0.05 | 0.00 | 0.00276 | 0.866 | [1, 1, 1] |
| oscillation | 0.1 | 0.00 | 0.00984 | 0.981 | [1, 1, 1] |
| oscillation | 0.2 | 0.00 | 0.0327 | 0.992 | [1, 1, 1] |
| oscillation | 0.4 | 0.00 | 0.0726 | 0.985 | [1, 1, 1] |
| oscillation | 0.8 | 0.00 | 0.119 | 0.982 | [1, 1, 1] |
| signed_residue | 0 | 1.00 | 0.000636 | 0.0607 | [1, 1, 1] |
| signed_residue | 0.01 | 0.00 | 0.00224 | 0.742 | [1, 1, 1] |
| signed_residue | 0.025 | 0.00 | 0.00612 | 0.936 | [3, 1, 1] |
| signed_residue | 0.05 | 0.00 | 0.0108 | 0.961 | [2, 3, 3] |
| signed_residue | 0.08 | 0.00 | 0.0177 | 0.978 | [2, 3, 3] |
| signed_residue | 0.13 | 0.00 | 0.0378 | 0.986 | [1, 3, 3] |

Route pass: **True**.

This is an empirical boundary scan with three repeats per setting, not a calibrated
type-I/type-II error guarantee.

# Refined near-boundary refusal scan

| Family | Violation strength | Accept fraction | Median validation RMSE | Median |lag-1| | Selected ranks |
|---|---:|---:|---:|---:|---|
| oscillation | 0 | 1.00 | 0.000582 | 0.105 | [1, 1, 1] |
| oscillation | 0.01 | 1.00 | 0.000625 | 0.0843 | [1, 1, 1] |
| oscillation | 0.02 | 1.00 | 0.000729 | 0.164 | [1, 1, 1] |
| oscillation | 0.03 | 1.00 | 0.00109 | 0.399 | [1, 1, 1] |
| oscillation | 0.04 | 0.00 | 0.0018 | 0.689 | [1, 1, 1] |
| oscillation | 0.05 | 0.00 | 0.00276 | 0.866 | [1, 1, 1] |
| signed_residue | 0 | 1.00 | 0.000636 | 0.0607 | [1, 1, 1] |
| signed_residue | 0.001 | 1.00 | 0.000617 | 0.107 | [1, 1, 1] |
| signed_residue | 0.0025 | 1.00 | 0.00084 | 0.209 | [1, 1, 1] |
| signed_residue | 0.005 | 1.00 | 0.00124 | 0.438 | [1, 1, 1] |
| signed_residue | 0.0075 | 0.00 | 0.0018 | 0.62 | [1, 1, 1] |
| signed_residue | 0.01 | 0.00 | 0.00224 | 0.742 | [1, 1, 1] |

Route pass: **True**.

The transition is empirical and depends on the declared noise, horizon, sampling,
candidate-rank cap, and diagnostic thresholds.

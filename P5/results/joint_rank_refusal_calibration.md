# Joint rank-selection and refusal calibration

Horizon: `14.0`; noise standard deviation: `0.0006`.

| Family | Level | Trials | Refusal fraction (Wilson 95%) | Elevated-rank fraction | Selected ranks |
|---|---|---:|---:|---:|---|
| oscillation | above | 10 | 1.00 [0.722, 1.000] | 0.00 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| oscillation | below | 10 | 0.10 [0.018, 0.404] | 0.00 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| oscillation | zero | 10 | 0.00 [0.000, 0.278] | 0.00 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| signed_residue | above | 10 | 1.00 [0.722, 1.000] | 0.10 | [1, 1, 1, 2, 1, 1, 1, 1, 1, 1] |
| signed_residue | below | 10 | 0.40 [0.168, 0.687] | 0.00 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| signed_residue | zero | 10 | 0.00 [0.000, 0.278] | 0.00 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |

Route pass: **True**.

All above-boundary refusals were triggered by residual correlation. The sole selected
rank-two case also failed the conditioning gate; no rank-three cap refusal occurred.

Ten repeats per setting narrow the route uncertainty but do not provide a final
uniform error guarantee over noise, horizon, and candidate-rank choices.

# Joint horizon/noise effect isolation

Horizon sweeps preserve the reference sampling interval; noise sweeps use H=14.

| Family | Level | Point | H | Noise | N(t) | Mismatch/noise | Refusal (Wilson 95%) | Elevated |
|---|---|---|---:|---:|---:|---:|---:|---:|
| oscillation | above | horizon_10 | 10 | 6.0e-04 | 70 | 5.82 | 1.00 [0.61, 1.00] | 0.00 |
| oscillation | above | horizon_14 | 14 | 6.0e-04 | 97 | 4.38 | 1.00 [0.61, 1.00] | 0.00 |
| oscillation | above | horizon_18 | 18 | 6.0e-04 | 124 | 3.85 | 0.17 [0.03, 0.56] | 0.00 |
| oscillation | above | noise_4.0e-04 | 14 | 4.0e-04 | 97 | 6.57 | 1.00 [0.61, 1.00] | 0.00 |
| oscillation | above | noise_9.0e-04 | 14 | 9.0e-04 | 97 | 2.92 | 0.17 [0.03, 0.56] | 0.00 |
| oscillation | zero | horizon_10 | 10 | 6.0e-04 | 70 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| oscillation | zero | horizon_14 | 14 | 6.0e-04 | 97 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| oscillation | zero | horizon_18 | 18 | 6.0e-04 | 124 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| oscillation | zero | noise_4.0e-04 | 14 | 4.0e-04 | 97 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| oscillation | zero | noise_9.0e-04 | 14 | 9.0e-04 | 97 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| signed_residue | above | horizon_10 | 10 | 6.0e-04 | 70 | 10.34 | 0.50 [0.19, 0.81] | 0.00 |
| signed_residue | above | horizon_14 | 14 | 6.0e-04 | 97 | 11.78 | 1.00 [0.61, 1.00] | 0.00 |
| signed_residue | above | horizon_18 | 18 | 6.0e-04 | 124 | 9.68 | 1.00 [0.61, 1.00] | 0.17 |
| signed_residue | above | noise_4.0e-04 | 14 | 4.0e-04 | 97 | 17.67 | 1.00 [0.61, 1.00] | 0.00 |
| signed_residue | above | noise_9.0e-04 | 14 | 9.0e-04 | 97 | 7.86 | 0.33 [0.10, 0.70] | 0.00 |
| signed_residue | zero | horizon_10 | 10 | 6.0e-04 | 70 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| signed_residue | zero | horizon_14 | 14 | 6.0e-04 | 97 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| signed_residue | zero | horizon_18 | 18 | 6.0e-04 | 124 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| signed_residue | zero | noise_4.0e-04 | 14 | 4.0e-04 | 97 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |
| signed_residue | zero | noise_9.0e-04 | 14 | 9.0e-04 | 97 | 0.00 | 0.00 [0.00, 0.39] | 0.00 |

Route pass: **False**.

These local one-factor sweeps isolate directional effects around the reference
operating point; they do not establish a global monotonicity theorem.

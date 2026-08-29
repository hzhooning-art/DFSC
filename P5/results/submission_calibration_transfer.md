# Submission calibration and noise-transfer audit

Frozen threshold: `0.49084813` (AR(1) calibration only).

| Noise generator | Separated support | 95% Wilson | Coalesced refusal | 95% Wilson |
|---|---:|---:|---:|---:|
| iid_gaussian | 40/40 | [0.912, 1.000] | 40/40 | [0.912, 1.000] |
| ar1 | 40/40 | [0.912, 1.000] | 40/40 | [0.912, 1.000] |
| ar2 | 40/40 | [0.912, 1.000] | 40/40 | [0.912, 1.000] |
| heteroscedastic | 40/40 | [0.912, 1.000] | 40/40 | [0.912, 1.000] |

Claim boundary: The frozen threshold is design conditional and is not a universal identifiability boundary.

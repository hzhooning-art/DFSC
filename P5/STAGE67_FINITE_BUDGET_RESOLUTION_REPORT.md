# Stage 67: Finite-Budget Spectral Resolution

## Objective

Test whether a classical multichannel matrix-pencil order selector is
sufficient for a rank-two mechanism claim, and identify finite observation
budgets under which the P5 evidence contract supports that claim.

## Design

- Positive two-rate signals with six grouped channels.
- Horizons: 4, 8, and 16.
- Samples per channel: 24 and 48.
- Relative noise standard deviations: 0.001, 0.005, and 0.015.
- Adjacent log-rate gaps: 0.02, 0.08, and 0.32.
- Twenty independent seeds in each of 54 cells (1,080 trials).
- A recovery is inaccurate when the maximum matched absolute log-rate error is
  greater than 0.20.

The comparator removes channel offsets by first differences, concatenates
channel Hankel blocks, and uses sequential BIC for rank one versus rank two.
The evidence gate additionally requires a normalized local-boundary index of
at least 0.49085, fitted rate ratio of at least 1.2, and agreement across three
Hankel aspect ratios with maximum log-rate standard deviation at most 0.15.

## Main result

| Quantity | Result |
|---|---:|
| Design cells / trials | 54 / 1,080 |
| Matrix-pencil BIC rank-two selections | 157 |
| Inaccurate BIC rank-two selections | 60 |
| Evidence-gated rank-two supports | 74 |
| Inaccurate evidence-gated supports | 0 |

Within the tested grid, a supported budget was found only for noise 0.001 and
log-rate gap 0.32. The smallest such budget used horizon 8 and 24 samples per
channel. No supported cell was found at the two larger noise levels or two
smaller gaps.

## Interpretation

BIC selection from a standard spectral estimator is not enough to justify a
two-rate mechanism interpretation. Cross-pencil stability removes unstable
pole estimates that can still receive an information-criterion preference.
The resulting atlas turns refusal into a finite-budget design decision: extend
the horizon or improve signal quality when possible; otherwise retain a lower
rank or report `INDETERMINATE`.

## Claim boundary

The zero observed inaccurate gated supports is conditional on this generator,
uniform sampling, Gaussian noise, amplitude distribution, thresholds, and
matrix-pencil implementation. It is not a zero-risk guarantee or a universal
spectral-resolution boundary.

## Reproduction

```bash
python P5/experiments/probe_finite_budget_resolution_atlas.py
python -m unittest P5.tests.test_matrix_pencil_resolution -v
```

The complete records are in
`P5/results/finite_budget_resolution_atlas.json`.

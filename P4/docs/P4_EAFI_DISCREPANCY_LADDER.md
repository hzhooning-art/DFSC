# P4 EAFI Discrepancy-Ladder Decision

## Calibrated test

An empirical 95% threshold was calibrated from 100 clean Mittag-Leffler
replicates. The null rejection rate was 5%, as expected. The test then used 100
independent replicates for each contamination level.

| contamination fraction | mean normalized score | rejection rate |
|---:|---:|---:|
| 0.00 | 0.936 | 0.03 |
| 0.05 | 0.888 | 0.02 |
| 0.10 | 0.839 | 0.00 |
| 0.20 | 0.724 | 0.00 |
| 0.30 | 0.638 | 0.00 |
| 0.50 | 0.645 | 0.00 |
| 1.00 | 24.681 | 1.00 |

The raw result is stored in
`P4/results/p4_eafi_discrepancy_ladder.json`.

## Exit decision

The predefined route-retention rule required rejection above 0.50 by 20%
contamination while keeping the null rejection rate at or below 0.10. The null
condition passed, but the sensitivity condition failed decisively. The mixed
signals can even reduce the best-fit residual because the contaminated signal
remains close to the fitted family over the finite observation window.

## Consequence for P4

The structural-mismatch-detector route is **closed as a primary P4 claim**. It
should not be rescued by increasing the error budget, because that would make
acceptance more permissive. The remaining defensible P4 contribution is the
narrower one: empirical error-aware profile inference for differentiable
Mittag-Leffler parameter estimation, with explicit coverage/area diagnostics.

That narrower contribution is still worth one further validation stage only if
it is framed as an uncertainty-aware inverse primitive rather than an automatic
model-selection system. If off-grid, unknown-noise, and external-family tests
cannot show useful coverage-area trade-offs, P4 should be stopped as an
independent paper and its results folded into a software or supplement track.

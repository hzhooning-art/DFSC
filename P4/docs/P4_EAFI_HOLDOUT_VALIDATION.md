# P4 EAFI Hold-out Validation

## Purpose

This experiment tests whether the EAFI profile-set rule can account for the
numerical floor of a low-order Mittag-Leffler evaluator without reusing the
same random replicates for calibration and evaluation.

## Protocol

- Forward family: `E_alpha(-rate * t^alpha)`.
- Reference evaluator: 100 series terms; tested evaluators: 6, 8, and 12 terms.
- Parameter grid: 27 alpha values by 26 rate values, 702 cells.
- Observation times: 32 points on `[0.05, 1.5]`.
- Noise levels: `sigma in {0.005, 0.01, 0.02}`.
- Calibration seeds: 40 seeds (`41000--41039`).
- Independent validation seeds: 40 seeds (`42000--42039`).
- Candidate multipliers: `{0.25, 0.5, 1, 2, 3}`.
- Selection rule: narrowest calibration profile set with empirical coverage at least 0.90.
- Review rule: validation coverage at least 0.90 and mean normalized profile area no more than 0.25.

The threshold is the sum of a noise term and a numerical-floor term,
`tau = 2*n*sigma^2 + c*n*floor_error`. Coverage is the fraction of
independent validation replicates whose true grid cell is retained. The area
fraction is the number of retained grid cells divided by 702.

## Results

| evaluator terms | noise | selected `c` | validation coverage | mean area fraction | alpha width | rate width | status |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 6 | 0.005 | 0.25 | 1.00 | 0.1002 | 0.2020 | 0.0600 | pass |
| 6 | 0.010 | 0.25 | 1.00 | 0.1446 | 0.2433 | 0.0825 | pass |
| 6 | 0.020 | 0.25 | 1.00 | 0.2636 | 0.2600 | 0.1315 | review |
| 8 | 0.005 | 0.25 | 1.00 | 0.0233 | 0.0778 | 0.0200 | pass |
| 8 | 0.010 | 0.25 | 1.00 | 0.0686 | 0.1568 | 0.0545 | pass |
| 8 | 0.020 | 0.25 | 1.00 | 0.2256 | 0.2583 | 0.1185 | pass |
| 12 | 0.005 | 0.25 | 1.00 | 0.0219 | 0.0735 | 0.0200 | pass |
| 12 | 0.010 | 0.25 | 1.00 | 0.0665 | 0.1550 | 0.0530 | pass |
| 12 | 0.020 | 0.25 | 1.00 | 0.2243 | 0.2578 | 0.1170 | pass |

The raw machine-readable result is
`P4/results/p4_eafi_holdout_validation.json`.

## Interpretation

1. The hold-out result supports the feasibility of an error-aware profile set
   for this controlled, known-propagator family. Coverage did not collapse when
   the random seeds were changed.
2. This is not a rigorous confidence guarantee. Coverage is empirical, the
   parameter grid is finite, and the noise model is Gaussian with known sigma.
3. The 6-term, `sigma=0.02` case fails the predefined area rule: the method
   protects coverage by retaining too much of the parameter grid.
4. The multiplier is not yet identifiable: `c=0.25` already gives 100%
   empirical coverage in every tested setting. A harder calibration design is
   required before freezing `c`, including unknown-noise estimation, off-grid
   truth, and near-miss structural models.

## Gate decision

**Partial pass.** The EAFI route remains viable as an independent P4 direction,
but it is not yet publication-ready. The follow-up stress test is recorded in
`P4/docs/P4_EAFI_ROBUSTNESS.md`: off-grid coverage survives, but a 10% near-miss
is accepted by both ordinary and EAFI profiles. The next method-development
step therefore needs a separate calibrated model-discrepancy statistic;
increasing the numerical error budget alone is not a solution.

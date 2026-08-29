# Stage 62 public-data protocol

This stage is frozen before fitting the public observations.

## Independent task

- Dataset: *Stress Relaxation Test Dataset of Cylindrical PVA Gel Polymer
  Electrolyte (GPE) Samples*.
- Record: Zenodo 21333840, DOI `10.5281/zenodo.21333840`.
- Access and license: open, CC BY 4.0 (`metadata.license.id` in the retrieved Zenodo record).
- Observations: three independently prepared specimens, each measured in three
  loading-relaxation cycles.
- Separation from prior P5 evidence: the measured force-time curves are fitted
  directly. They are not used as residual backgrounds for a synthetic model.

## Frozen preprocessing

For each cycle, the relaxation segment begins 0.25 s after the first sample at
which displacement reaches 99.9% of its cycle maximum. Time is re-zeroed and
force is divided by its first retained value. No smoothing is applied.

## Candidate realization

Each curve is represented as

`y_c(t) = b_c + sum_j a_cj exp(-r_j t)`,

where amplitudes and baselines are non-negative and rates are shared across
curves. Candidate ranks are 1, 2, and 3. Rates are estimated on two specimens;
the held specimen supplies an external prediction check. On the held specimen,
curve-specific amplitudes are calibrated on the first 60% of the declared
window and the remaining 40% is predicted.

## Evidence and refusal rule

A transition from rank `m-1` to rank `m` is accepted only when all gates pass:

1. pooled training BIC improves by at least 10;
2. median held-specimen late-window NRMSE improves by at least 5%;
3. the maximum across-fold standard deviation of matched log-rates is at most
   0.50;
4. adjacent fitted rates differ by a factor of at least 1.25;
5. all numerical solves are finite.

If BIC favors a higher rank but any predictive or stability gate fails, the
output is `INDETERMINATE`, not a forced mechanism label. Thresholds may not be
changed after inspecting Stage 62 results.

## Boundary audit

The same frozen rule is evaluated over declared observation horizons of 4, 8,
15, and 28 seconds and per-curve sample budgets of 12, 24, 48, and 96. The map
is descriptive: it identifies where the present observations support a rank,
where they are insufficient, and where numerical failure occurs.

# P4 EAFI Feasibility Result

This is the first feasibility gate for the replanned independent P4 paper,
**Error-Aware Fractional Operator Identification under Structural Mismatch**.

## Known-truth inverse test

Synthetic observations were generated from
`E_alpha(-lambda t^alpha)` with `alpha=0.78`, `lambda=0.65`, 32 observation
times, and Gaussian noise `sigma=0.01`. The reference used 100 evaluator
terms. The ordinary inverse profile used a deliberately low four-term
evaluator, while the error-aware profile inflated its acceptance threshold by
the observed value discrepancy between four and 100 terms.

| Procedure | Coverage | Mean accepted grid points |
|---|---:|---:|
| Ordinary profile | 0/20 = **0%** | 36.35 |
| Error-aware profile | 20/20 = **100%** | 702.00 |

The order gradient remained finite, with absolute magnitude `1.5154`.

## Structural-mismatch screening

The observations were generated from a strongly misspecified
stretched-exponential law. The best fractional-model loss divided by the
nominal noise loss had mean `99.29` and minimum `93.70`. Under the provisional
threshold of 3, ordinary acceptance was `0%` and EAFI abstention was `100%`.

## Interpretation

The result supports the **mechanistic feasibility** of P4: numerical error can
materially change inverse interval coverage, and a mismatch score can trigger
refusal. However, the error-aware interval is currently far too conservative,
covering the full tested grid. This is not yet a publishable calibration
result.

## Required next gate

The first calibration/hold-out gate is recorded in
`P4/docs/P4_EAFI_HOLDOUT_VALIDATION.md`. It gives a partial pass: coverage is
stable on independent seeds, but the multiplier is not yet identifiable because
the tested configurations all achieve 100% empirical coverage. The next
method-development experiment must use off-grid truth, unknown noise, and
near-miss structural models before a final P4 claim is made.

Result file: `P4/results/p4_eafi_feasibility.json`.

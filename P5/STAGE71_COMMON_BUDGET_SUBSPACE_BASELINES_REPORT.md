# Stage 71: Common-Budget Subspace and Prony Order Baselines

Updated: 2026-09-03

## Objective

Compare the frozen power-certified detector with signal-processing baselines
that issue an explicit rank-one or rank-two decision on exactly the same 1,152
Stage 69 evaluation records. Existing P5 Prony and NNLS prediction errors are
not reused as order decisions.

The added full-coverage comparators are shared linear-prediction/Prony AICc and
BIC, plus block-Hankel covariance-eigenvalue AIC and MDL. Matrix-pencil AICc is
retained. Every method receives the same six grouped channels, horizon,
sampling budget, noise realization, and seed.

## Aggregate result

| Method | Coverage | Overall accuracy | Selective risk | False elevation | False reduction | Rank-two detection | Median runtime (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matrix-pencil AICc | 1.0000 | 0.6441 | 0.3559 | 0.0000 | 0.5339 | 0.4661 | 0.8273 |
| Block-Hankel AIC | 1.0000 | 0.6658 | 0.3342 | 0.6771 | 0.1628 | 0.8372 | 0.1726 |
| Block-Hankel MDL | 1.0000 | 0.6146 | 0.3854 | 0.2839 | 0.4362 | 0.5638 | 0.1606 |
| Shared Prony AICc | 1.0000 | 0.5035 | 0.4965 | 0.6224 | 0.4336 | 0.5664 | 0.0778 |
| Shared Prony BIC | 1.0000 | 0.5017 | 0.4983 | 0.5911 | 0.4518 | 0.5482 | 0.0685 |
| Power-certified selective | 0.2943 | 0.2535 | 0.1386 | 0.0000 | 0.0612 | 0.2135 | 3.3126 |

## Interpretation

No full-coverage method dominates both error directions. Block-Hankel AIC has
the highest unconditional accuracy and rank-two detection, but labels 67.71%
of true rank-one cases as rank two. Matrix-pencil AICc eliminates observed
false elevation but reduces 53.39% of rank-two cases to rank one. The two
shared-Prony criteria are near chance overall and strongly over-elevate rank.

The power-certified method has much lower error among issued decisions and
controls both directional errors, but abstains on 70.57% of records. Therefore
the supported contribution is a distinct reliability--coverage operating
regime, not higher unconditional classification accuracy or lower cost.

The noise split is material. Under white noise, block-Hankel AIC false elevation
is 98.96%; under AR(1) it is 36.46%. Matrix-pencil AICc instead has false
reduction of 69.27% under white noise and 37.50% under AR(1). The candidate's
false reduction is 7.03% and 5.21%, respectively, with zero observed false
elevation in both families.

## Methodological boundary

The block-Hankel criteria use overlapping columns as effective snapshots.
Those columns are dependent, so AIC/MDL are classical-style subspace
comparators rather than exact independent-snapshot likelihood calculations.
This limitation is part of the comparison and must remain visible in the
manuscript.

Stage 71 fills the missing same-budget order-baseline table. It does not yet
demonstrate transfer of the power certificate to measured signals. Stage 72
must replay the frozen rule on held-out PVA/copper/public tasks without using
their outcomes to modify thresholds.

## Reproduction

```console
python P5/experiments/probe_common_budget_subspace_baselines.py --stdout-summary
python -m unittest P5.tests.test_common_budget_subspace_baselines -v
```

The aggregate artifact is `P5/results/common_budget_subspace_baselines.json`.

# Stage 64: third public domain and identifiability boundary

## Completed scope

This stage adds an independent hydraulic-system application and a formal local
indistinguishability result to the P5 evidence-gated memory-realization study.
The new domain is independent of the PVA stress-relaxation and gas-sensor tasks
in both physical process and experimental grouping unit.

## Frozen hydraulic protocol

- Source: UCI Condition Monitoring of Hydraulic Systems, DOI
  `10.24432/C5CW21`.
- Grouping unit: one independent 60-second load cycle; sensor channels are not
  treated as independent replicates.
- Fixed operating state: valve `100`, pump leakage `0`, accumulator `130`, and
  stable flag `0`.
- Cooler states: `3`, `20`, and `100`; ten cycles per state.
- Channels: `CE`, `CP`, `TS1`, and `TS2`.
- Analysis window: seconds `20--55`; normalization uses only the first 22
  samples and applies no smoothing.

## Main result

The final decision is **INDETERMINATE**, and all 27 predeclared threshold
combinations return the same decision.

| Rank | Mean BIC | Median held-cycle NRMSE | Local boundary index |
|---:|---:|---:|---:|
| 1 | -8509.645 | 0.23780 | not applicable |
| 2 | -9167.872 | 0.51128 | 0.057112 |
| 3 | -9051.191 | 0.42922 | 1.84865e-07 |

The rank-two model improves in-sample BIC but worsens held-cycle prediction.
Its median relative improvement over rank one is `-0.73537`, with a
cycle-cluster bootstrap 95% interval `[-1.4400, -0.38405]`. The additional
rates also coalesce at a numerical boundary. This is a direct refusal case:
information gain alone does not support an interpretable extra mode.

## Formal result

The new theorem considers adjacent exponential rates under independent
Gaussian observations. For the two-mode and merged one-mode experiments it
defines

\[
R_n^2 = \frac{\delta_n^2}{\sigma^2}
\sum_{i,\ell} b_i^2 t_{i\ell}^2
\exp\{-2\min(\lambda,\lambda+\delta_n)t_{i\ell}\}.
\]

The Kullback--Leibler divergence is at most `R_n^2/2`. Consequently, if
`R_n -> 0`, every rank test has type-I plus type-II error tending to one, so no
uniformly consistent rank decision exists along that sequence. A consistent
diagnostic with a shrinking threshold therefore refuses consistently inside
the indistinguishable region; under fixed separation, local identifiability,
and consistent remaining gates it accepts the true rank.

The claim is deliberately local and Gaussian. It does not turn correlated
noise, model misspecification, or an empirical certificate into a global
theorem.

## Artifacts and verification

- Frozen protocol: `STAGE64_HYDRAULIC_PROTOCOL.md`.
- Experiment: `experiments/probe_public_uci_hydraulic_transients.py`.
- Machine-readable result: `results/public_uci_hydraulic_transients.json`.
- Human-readable result: `results/public_uci_hydraulic_transients.md`.
- Figure: `figures/fig_stage64_hydraulic_transients.pdf` and `.png`.
- Standalone theory note: `THEOREM_IDENTIFIABILITY_BOUNDARY.md`.
- English and Chinese manuscripts: `paper/manuscript_en.pdf` and
  `paper/manuscript_zh.pdf`.
- Test suite: 114 tests passed.
- PDF verification: both manuscripts compile to ten pages with no overfull
  boxes, undefined references, or BibTeX warnings; theorem and application
  pages were visually inspected.

# Stage 63 frozen public gas-recovery protocol

## Question

Can a shared finite-dimensional relaxation realization be supported, or must it
be refused, across independent gas-sensor experiments when prediction is tested
on unseen acquisition batches?

## Frozen design

- Source: UCI dataset 308 (`10.24432/C5BG7G`), CC BY 4.0.
- Independent units: 50 non-air gas exposures; channels within an exposure are
  clustered and are not treated as independent replicates.
- Recovery window: 180--300 s, registered by the public acquisition protocol.
- Preprocessing: median within each 12 s ventilator cycle; no fitted smoothing;
  normalize by the first recovery-cycle response without using held-out tail data.
- Cross-validation: leave one acquisition batch out (five folds).
- Candidate ranks: 1, 2, 3 shared positive decay rates with channel-specific
  signed amplitudes and intercepts.
- Calibration/prediction split: first 60% / final 40% of recovery cycles.
- Gates: BIC improvement >= 10; held-tail NRMSE improvement >= 5%; maximum
  leave-batch log-rate standard deviation <= 0.8; adjacent rate ratio >= 1.2.
- Baselines: independent nonlinear exponential fit, fixed log-grid NNLS, and
  Prony linear recurrence under the identical early/late split.
- Primary uncertainty: experiment-cluster bootstrap and experiment-level paired
  Wilcoxon test. Sensor channels are summarized within experiment first.
- Output includes `INDETERMINATE`; it never forces a mechanistic label.

This protocol was written before fitting the Stage 63 task.

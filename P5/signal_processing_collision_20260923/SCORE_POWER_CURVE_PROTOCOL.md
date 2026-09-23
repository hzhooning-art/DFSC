# Frozen moderate-signal power-curve protocol

Date: 2026-09-10. Freeze before simulation.

Use C=6, n=24, times on [0,8], lambda=.4, p=.5, S=10000, independent
Gaussian sample-mean noise and an independent pooled scale exactly as in the
finite-local-limit study. Recalculate the diversity, crossover, and
proportional path constants for target full-residual noncentralities 4, 8, 12,
and 18. Evaluate m=4096 and 16384 with 1024 trials per cell. Use deterministic
cell seeds `6091050000000 + 100000*j + trial`.

On every realization apply both the phase-matched C-dimensional score F test
and the omnibus union-residual F test at alpha=.05. Neither threshold may be
tuned. Predict score power from its population score projection and predict
omnibus power from its population distance using their respective local
noncentral-F laws.

Predeclared gates at m=16384:

1. score empirical-prediction error is at most .05 in every cell;
2. omnibus empirical-prediction error is at most .06 in every cell;
3. score power exceeds omnibus power by at least .12 for every target and path;
4. score-space noncentrality retains at least 92% of full-distance
   noncentrality in every cell;
5. all values are finite probabilities and all amplitudes remain positive.

This protocol tests whether the earlier power gain persists away from the
near-one power ceiling. It remains a fixed-design local experiment.

# Frozen validation of the phase-matched tangent score test

Date: 2026-09-10. Freeze before simulation.

Use the exact C=6, n=24 design, signal parameters, three local paths, null
path, m values, Gaussian sample means and independent chi-square scale model
from `FINITE_LOCAL_LIMIT_PROTOCOL.md`. Use its seeds and trial counts: 1024
per alternative cell and 2048 per null cell.

For every realization, fit the one-rate null over [0.2,0.8]. At the fitted
rate and fitted merged-amplitude vector, form the `(C-1)` diversity score
directions and one curvature score direction in
`PHASE_MATCHED_SCORE_THEORY.md`. Reject when T/C exceeds the 0.95 quantile of
F(C,nu). No threshold tuning or finite-sample correction is allowed.

For every population alternative, compute the noncentrality obtained by
projecting its noiseless mean onto the plug-in score space at the closest null
fit. Compare empirical power with the corresponding noncentral-F prediction.

Predeclared gates:

1. null rejection is at most .07 at every m;
2. at m=16384, absolute error from the exact-projection noncentral-F prediction
   is at most .05 on all three paths;
3. at m=16384, score-test power exceeds the union-residual power from the
   frozen finite-local-limit study by at least .15 on every path;
4. exact-projection score noncentrality differs from the full profiled-distance
   noncentrality by at most 8% at m=16384;
5. all fitted rates, score norms and statistics are finite, and the fitted
   merged-amplitude norm is positive.

Passing demonstrates the predicted local efficiency in one frozen design. It
does not prove finite-sample size control for the plug-in score test.

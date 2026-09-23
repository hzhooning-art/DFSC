# Frozen fully matched likelihood benchmark

Date: 2026-09-10. Freeze before outcomes.

The comparator is an oracle Monte-Carlo-calibrated generalized likelihood
ratio for exactly the P5 observation model: C=6 multichannel curves, free
channel intercepts and amplitudes, one shared real rate under the null, two
shared real rates under the alternative, Gaussian sample-mean noise, and an
independent pooled variance estimate. This gives the comparator the true null
rate and amplitude configuration for calibration; the proposed tests do not
receive these quantities.

Use n=24 times on [0,8], rate interval [.2,.8], lambda=.4, p=.5, S=10000,
m=16384, and the diversity, crossover and proportional paths at target
noncentralities 4 and 8. Use 2048 null simulations to set the empirical .95
GLR quantile and a disjoint 2048 null simulations to measure size. Use 512
alternative trials per cell. Seeds begin at 6091070000000 for calibration,
6091080000000 for null evaluation, and 6091090000000 plus 100000 per
alternative cell.

The rank-two fit profiles all linear coefficients. Search rate pairs using a
41-point triangular grid including the coalescent derivative closure, followed
by bounded two-dimensional refinement from the best grid point. On the first
and last realization of each cell, repeat with an 81-point grid; require GLR
differences at most .05. Apply phase-matched and cone-matched score tests to the
same realizations with their analytic .05 critical values.

Predeclared gates:

1. held-null GLRT rejection is in [.03,.07];
2. all replay 41-vs-81 grid GLR differences are at most .05;
3. optimizer RSS2 never exceeds RSS1 by more than 1e-9 and all outputs are
   finite;
4. in every alternative cell, neither matched GLRT nor cone score is declared
   dominated solely from Monte Carlo noise: report paired differences and
   Wilson intervals without a superiority gate.

SAMP remains a neighboring single-sequence complex-exponential method rather
than a fully matched comparator. Common-pole matrix pencil and Hankel criteria
remain secondary algorithmic baselines from the earlier frozen study.

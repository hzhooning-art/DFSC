# Frozen finite-m local-limit validation

Date: 2026-09-10, frozen before simulation.

Use C=6, n=24 equally spaced times on [0,8], K=[.2,.8], lambda=.4,
p=.5, S=10000, sigma=1, and target asymptotic noncentral parameter 36.
The merged amplitude direction is constant and positive. The orthogonal diversity
direction has two opposite entries; all component amplitudes remain positive.

Compare three local paths:

1. diversity: B/S=.04 and delta=kappa*m^(-1/2);
2. crossover: B/S=.04*m^(-1/2) and delta=kappa*m^(-1/4);
3. proportional: B=0 and delta=kappa*m^(-1/4).

Choose each kappa once from the leading V,W formula so its limiting noncentrality
is 36. Use m=256,1024,4096,16384. For each alternative cell use 1024 trials;
for each null m use 2048 trials. Generate the sufficient sample mean directly
with N(0,1/m) entry noise and draw the independent within-repeat scale from
chi-square with nu=C*n*(m-1). This is distributionally identical to retaining
the mean and pooled variance from m raw matrices.

For every dataset minimize the exact signed one-rate residual over K. Use an
801-point rate grid, refine the best adjacent interval with bounded scalar
minimization, and include both endpoints. Reject with the finite-sample union F
threshold using d=C(n-2). Record the exact noiseless profiled distance, empirical
rejection, the finite-nu noncentral-F prediction using that distance, the limiting
target prediction, and a KS distance to the exact-distance local approximation.

Predeclared diagnostics:

- at m=16384, each alternative path has exact noncentrality within 8% of 36;
- empirical rejection differs from its exact-distance local prediction by <=.06;
- KS distance at m=16384 is <=.07;
- every null rejection frequency is <=.07;
- no numerical minimization failure or nonfinite statistic.

These gates diagnose convergence in one frozen design. Passing does not prove a
uniform Berry-Esseen bound, validate other designs, or turn the asymptotic power
curve into a finite-sample theorem.

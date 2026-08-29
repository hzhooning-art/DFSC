# Minimal theory for shared-memory identification and refusal

## 1. Local identifiability

Let observations satisfy `y = f(theta) + epsilon` on the declared sample grid,
and let `J(theta)` be the Jacobian of `f` with respect to the shared rates and
nuisance parameters. A necessary condition for local parameter identifiability
is full column rank of `J`. Under independent Gaussian noise, `J^T J` is the
local Fisher-information matrix up to the noise variance; with correlated noise
it is replaced by `J^T Sigma^{-1} J`. A small minimum singular value therefore
signals weak local distinguishability even when the fitted trajectory error is
small.

## 2. Why nearby exponential rates must be refused

For positive rates `lambda` and `mu` on `[0,T]`, the mean-value theorem gives

`|exp(-lambda t) - exp(-mu t)| <= |lambda-mu| t exp(-min(lambda,mu)t)`.

Hence the two design columns become collinear as `|lambda-mu| -> 0`; the
smallest singular value of the exponential design tends to zero. Additional
nominal rank can then reduce residual error while failing to identify an
additional timescale. The minimum adjacent-rate-ratio gate operationalizes this
necessary separation check. It is deliberately conservative and is not a
universal hypothesis test.

## 3. Why more samples need not yield a monotone empirical decision

For nested observations and known covariance, Fisher information is monotone in
the positive-semidefinite order. The P5 boundary experiment does not satisfy
that idealization: grids change with horizon, nuisance amplitudes are refitted,
nearby samples have correlated residuals, and nonlinear optimization has a
finite start budget. Thus raw sample count can rise while effective information
changes little, or while conditioning worsens. The protocol reports tail
coverage, an AR(1) effective-sample-size proxy, sensitivity conditioning, and
start-budget stability separately instead of treating sample count as evidence.

## 4. Statistical meaning of refusal

`INDETERMINATE` means that the predeclared evidence gates do not jointly support
a stable finite shared rank on the tested domain. It is not evidence that no
shared memory exists, and it does not assign a different physical mechanism.
Error control for a formal Type-I/Type-II test would require a calibrated null
law specific to the sampling process and noise covariance; the current gates
are auditable decision constraints rather than such a universal test.

## 5. Formal adjacent-rate boundary

For independent Gaussian observations, compare a model containing
`a_i exp(-lambda t) + b_i exp(-(lambda+delta_n)t)` with the lower-rank model
that merges the pair into `(a_i+b_i) exp(-lambda t)`. Define

`R_n^2 = delta_n^2 sigma^{-2} sum_{i,l} b_i^2 t_il^2 exp(-2 min(lambda,lambda+delta_n)t_il)`.

The mean-value theorem and the Gaussian KL formula give
`KL(P_r || P_{r-1}) <= R_n^2/2`. Pinsker's inequality then implies that if
`R_n -> 0`, every rank test has type-I plus type-II error tending to one. No
uniformly consistent rank decision is possible along this sequence.

The complete assumptions, proof, and consistent-refusal corollary are recorded
in `THEOREM_IDENTIFIABILITY_BOUNDARY.md`. The software field
`local_boundary_index` is a local empirical analogue formed from the smallest
projected log-rate sensitivity and the adjacent log-rate gap.

## Embedded multi-rate obstruction and scope

If all nuisance components and all decay rates except one adjacent pair are fixed, or are estimated with error smaller than the adjacent-rate separation scale, the rank-r and merged rank-(r-1) classes contain the two-point subexperiment used by the adjacent-rate theorem. Hence, as that separation scale tends to zero, no rank test can be uniformly consistent over the larger multi-rate classes.

This is a local or embedded impossibility result. It does not establish global identifiability for unrestricted nonlinear kernels, irregular observation designs, non-Gaussian noise, or jointly drifting nuisance parameters. Those settings require additional assumptions and separate analysis.

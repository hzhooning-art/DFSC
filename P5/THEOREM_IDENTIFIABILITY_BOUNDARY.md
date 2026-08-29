# Formal identifiability boundary for shared finite-memory rank

## Model and notation

For independent observations indexed by `(i,l)`, consider the Gaussian shared-rate model

`Y_il = c_i + sum_k a_ik exp(-lambda_k t_il) + epsilon_il`,

where `epsilon_il ~ N(0, sigma^2)`, `0 <= t_il <= T_n`, and amplitudes and offsets are nuisance parameters. Signed amplitudes are allowed. Let a rank-`r` alternative contain adjacent rates `lambda` and `lambda + delta_n`, with amplitudes `a_i` and `b_i`. Let the comparison rank-`r-1` model merge these terms into `(a_i+b_i) exp(-lambda t)`.

Define

`R_n^2 = (delta_n^2 / sigma^2) sum_{i,l} b_i^2 t_il^2 exp(-2 min(lambda,lambda+delta_n) t_il)`.

## Theorem: Gaussian adjacent-rate indistinguishability

Under the declared Gaussian model,

1. `KL(P_r,n || P_r-1,n) <= R_n^2 / 2`.
2. If `R_n -> 0`, every measurable rank test satisfies `alpha_n + beta_n -> 1` along this design sequence. No uniformly consistent procedure can distinguish the adjacent modes on that sequence.

## Proof

For each observation,

`d_il = b_i [exp(-(lambda+delta_n)t_il) - exp(-lambda t_il)]`.

The mean-value theorem yields

`|d_il| <= |b_i delta_n| t_il exp(-min(lambda,lambda+delta_n)t_il)`.

Gaussian measures with common covariance `sigma^2 I` satisfy

`KL(P_r || P_r-1) = ||d||_2^2 / (2 sigma^2) <= R_n^2/2`.

Pinsker's inequality gives `TV(P_r,P_r-1) <= R_n/2`. The two-point testing bound `alpha_n + beta_n >= 1-TV(P_r,P_r-1)` proves the second claim.

## Conditional corollary: calibrated refusal

Suppose a diagnostic `Rhat_n` on a fixed design obeys `|Rhat_n-R_n| = o_p(c_n)` for `c_n -> 0`, and its threshold is selected only from calibration data independent of evaluation data.

- If `R_n=o(c_n)`, the rule `Rhat_n <= c_n` outputs `INDETERMINATE` with probability tending to one.
- If adjacent rates remain fixed and separated, the projected information grows, parameter estimates are consistent, and the other evidence gates are consistent, the rule accepts the true rank with probability tending to one.

This corollary is conditional. It is not supplied by the impossibility theorem alone.

## Relation to the implementation

The implementation reports `local_boundary_index`, formed from the smallest log-rate sensitivity after projecting out curve-specific offsets and amplitudes, the minimum adjacent log-rate gap, and residual noise. It also reports `normalized_local_boundary_index`, which divides by the square root of the observation count to remove the leading sample-size scaling.

Neither field is an estimator of `R_n` without additional assumptions. Thresholds are calibrated on a declared semi-synthetic design and evaluated on disjoint seeds. Values are comparable within that declared task and normalization, not across arbitrary datasets.

## Scope

The theorem is a local two-point impossibility result for the declared independent Gaussian observation model. It does not prove global identifiability for arbitrary finite mixtures, colored noise, unknown registration, constrained amplitudes, or model misspecification. Correlated-residual sensitivity is audited separately using an AR(1) profile likelihood and an effective-sample-size diagnostic.

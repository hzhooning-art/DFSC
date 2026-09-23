# Phase-matched tangent score test

## 1. Motivation

The union-residual F test is finite-sample valid, but its local residual has
`C(n-2)-1` degrees of freedom. Only C directions carry the leading signal for
a coalescing second rate. Testing the entire residual space therefore pays a
large power cost unrelated to the P5 alternative.

Let `s=a+b`, and at a regular one-rate null define

\[
g_1={P_{\{1,e\}^\perp}e'\over\|P_{\{1,e\}^\perp}e'\|},\qquad
g_2={P_{\{1,e,e'\}^\perp}e''\over\|P_{\{1,e,e'\}^\perp}e''\|}.
\]

The phase-matched score space is

\[
\mathcal E(s,\lambda)
=\{u g_1^T:u\perp s\}\oplus
\operatorname{span}\{s g_2^T\}.
\]

It has dimension `(C-1)+1=C`. The two summands are orthogonal in channel
space or time space.

## 2. Leading signal identity

**Theorem 1 (phase-matched normal decomposition).** Under the compact interior
conditions of the uniform crossover theorem, the projection of a coalescing
two-rate mean onto the phase-matched score space at its closest regular
one-rate representation satisfies

\[
\|P_{\mathcal E}(M-\Pi_{\mathcal M_1}M)\|_F^2
=BV\delta^2+Sc^2W\delta^4
+O(B|\delta|^3+S|\delta|^5).
\]

Consequently, the C-dimensional score retains the full leading separation of
the much larger profiled residual.

**Proof.** The exact channel decomposition writes `b=ps+b_perp`, with
`b_perp` orthogonal to `s`. After profiling intercepts, amplitudes and the
first-order common rate displacement, the leading normal terms are

\[
\delta\,b_\perp(P_2e')^T
\quad\text{and}\quad
c\delta^2s(P_3e'')^T.
\]

The first belongs to the `(C-1)`-dimensional diversity summand and has squared
norm `BV delta^2`. The second belongs to the curvature summand and has squared
norm `S c^2 W delta^4`. They are orthogonal because `b_perp` is orthogonal to
`s` (and also because `g2` is orthogonal to `g1`). The uniform remainder is the
same localized remainder controlled in the crossover theorem. □

## 3. Feasible statistic

Fit the one-rate null to the sample mean, obtaining `(lambda_hat,s_hat)`, and
construct `g1_hat,g2_hat`. Put `s_unit=s_hat/||s_hat||`,

\[
z_1=\bar Y g_{1,\mathrm{hat}},\qquad
z_2=\bar Y g_{2,\mathrm{hat}},
\]

and

\[
T_m={m\over s^2}\left(
\|(I-s_{unit}s_{unit}^T)z_1\|^2
+(s_{unit}^Tz_2)^2\right).
\]

**Theorem 2 (null and local limits).** At an interior regular one-rate null,
with fixed C,n and a consistent null fit,

\[
T_m\Rightarrow\chi_C^2.
\]

If an independent scale estimate has `nu` chi-square degrees of freedom, then
`T_m/C` has limiting `F_{C,nu}` calibration. Along the P5 local alternatives,

\[
T_m\Rightarrow\chi_C^2(\rho^2),\qquad
\rho^2={m\over\sigma^2}
\{BV\delta^2+Sc^2W\delta^4\}+o(1).
\]

**Proof sketch.** Null least squares removes the tangent coordinates for
channel intercepts, channel amplitudes, and the common rate. The estimated
score space converges to the C-dimensional orthogonal space above. Gaussian
projection and Slutsky's theorem give the central limit. Under contiguous
alternatives, Theorem 1 supplies the mean shift and Le Cam's third-lemma form
of the same Gaussian projection gives the noncentral limit.

This result is asymptotic, unlike the conservative union-residual size bound.
A complete paper proof must make the plug-in score-space expansion explicit
and treat the rate-search endpoints separately.

## 4. Methodological contribution

The statistic is phase-adaptive without estimating whether diversity or
curvature dominates. It adds their orthogonal score energies and therefore
uses the same critical rule on the `m^-1/2`, crossover, and `m^-1/4` branches.
Relative to the omnibus residual test, it retains the same leading
noncentrality while changing the limiting null degrees of freedom from
`C(n-2)-1` to C.

This is a stronger method claim than the earlier controller: the dimension
reduction is derived from the joint-degeneracy geometry, rather than selected
as an engineering projection.

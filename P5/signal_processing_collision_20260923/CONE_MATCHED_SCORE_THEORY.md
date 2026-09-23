# Cone-matched phase score test

## 1. The extra structure

With ordered rates `lambda < lambda+delta`, nonnegative component amplitudes,
and `p` in `(0,1)`, the leading proportional-curvature displacement is

\[
c\delta^2s(P_3e'')^T,\qquad c=p(1-p)/2>0.
\]

After orienting `g2=P3 e''/||P3 e''||`, its scalar score has known nonnegative
sign. The diversity displacement remains an unknown vector in the
`(C-1)`-dimensional channel complement. The local alternative is therefore a
product cone

\[
\mathcal K=\mathbb R^{C-1}\times\mathbb R_+,
\]

rather than an unrestricted C-dimensional space.

## 2. Statistic and null law

Let `Z_D` be the standardized diversity score vector and `Z_Q` the oriented
curvature score. Define

\[
Q=\|Z_D\|^2+(Z_Q^+)^2,
\qquad Z_Q^+=\max(Z_Q,0).
\]

**Proposition 1 (chi-bar-square calibration).** At a regular null with known
variance,

\[
Q\Rightarrow {1\over2}\chi^2_{C-1}+{1\over2}\chi^2_C.
\]

With an independent variance estimate having `nu` degrees of freedom, the
limiting CDF at raw threshold q is

\[
{1\over2}F_{C-1,\nu}\!\left({q\over C-1}\right)
+{1\over2}F_{C,\nu}\!\left({q\over C}\right).
\]

**Proof.** Under the null, `Z_D`, `Z_Q`, and the independent scale are mutually
independent. On `Z_Q<0`, which has probability one half, the projection onto
the cone sets the last coordinate to zero and leaves chi-square `C-1`. On
`Z_Q>0`, the squared last coordinate is distributed as an unconditional
chi-square one variate and adds independently to the diversity energy, giving
chi-square C. Studentization produces the displayed F mixture. □

## 3. Local optimality interpretation

The statistic is the Gaussian likelihood-ratio projection statistic for the
local cone `K`. It is therefore the natural cone LRT in the phase-matched
limit experiment. Relative to the two-sided score test, it uses the known sign
of the quartic branch and has a smaller null critical value. The gain is
largest near proportional amplitudes, where the curvature coordinate carries
most separation; it should be modest on diversity-dominated paths.

The claim is local and requires ordered rates and inactive nonnegativity
constraints. If signed component amplitudes are allowed, the curvature sign is
not protected and the two-sided C-dimensional score test is the correct safe
version.

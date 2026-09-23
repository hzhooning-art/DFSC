# Local limiting power on the joint-degeneracy boundary

Date: 2026-09-10. This note fills the finite-rho gap left by the matched-rate
argument. The manifold limit is standard; the P5-specific content is the
substitution of the uniform joint-degeneracy expansion and its normal vectors.

## Regular null point

Let M0 be the signed shared-one-rate mean set in R^(C by n). Fix a null mean N0
whose rate is in the interior of K and whose C-vector of amplitudes is nonzero.
With at least three distinct times, M0 is a smooth embedded manifold near N0 of
dimension 2C+1: C baselines, C amplitudes, and one shared rate. Its codimension is

    r0 = Cn-(2C+1) = C(n-2)-1.                                  (1)

The zero-amplitude stratum and constrained rate boundaries are excluded.

## Local alternative formulation

For each m, let M_m be a two-rate mean and let N_m be its unique closest local
point in M0. Assume N_m->N0 and

    sqrt(m)(M_m-N_m) -> h,                                      (2)

where h lies in the normal space of M0 at N0. Let sigma_m->sigma>0. The profiling
first-order condition makes M_m-N_m normal at N_m; smoothness transfers the limit
to the normal space at N0. Moreover

    lambda_nc = ||h||_F^2/sigma^2
              = lim m D_m^2/sigma_m^2.                          (3)

Observe m iid signal matrices and let R_m=dist(Ybar,M0).

## Proposition: profiled residual limit

Under (1)-(3),

    m R_m^2/sigma_m^2  =>  chi-square_r0(lambda_nc).             (4)

This follows from the local tangent approximation of a smooth embedded manifold:
sqrt(m)(Ybar-N_m) converges to h+sigma Z, profiling removes the tangent component,
and the squared normal component has the stated noncentral chi-square law.

If an independent scale s_m^2 obeys nu_m s_m^2/sigma_m^2~chi-square_nu_m with
nu_m->infinity, replacing sigma_m by s_m leaves (4) unchanged. With fixed nu,
the corresponding limit after division by r0 s_m^2 is noncentral F(r0,nu,lambda_nc).

The asymptotically calibrated profile test rejects when

    m R_m^2/(r0 s_m^2) > F_{r0,nu_m;1-alpha}.                    (5)

It has limiting size alpha at regular null points and limiting local power

    pi(lambda_nc)=Pr{chi-square_r0(lambda_nc)
                      > chi-square_{r0;1-alpha}}                 (6)

when nu_m->infinity. This is pointwise over the regular stratum, not the uniform
finite-sample guarantee of the conservative union test.

## Relation to the exact-level union test

The exact-level union test uses d=C(n-2), one more residual degree of freedom,
because it evaluates a fixed true rate to obtain finite-sample domination and
does not subtract the fitted shared rate. As nu_m->infinity its rejection
threshold is chi-square_{d;1-alpha}, while its local statistic still has the
limit in (4). Its local power is therefore

    pi_union(lambda_nc)=Pr{chi-square_r0(lambda_nc)
                            > chi-square_{d;1-alpha}}.            (7)

It is slightly conservative locally, but retains uniform finite-sample size over
all rates including boundaries. Equation (7) is much sharper than the earlier
triangle-inequality sufficient-power bound; it is asymptotic and pointwise.

## Joint-degeneracy paths

Use the notation of UNIFORM_CROSSOVER_THEORY.md. Along any crossover sequence
for which the normalized channel direction also converges,

    lambda_nc = lim (m/sigma_m^2)
      [B_m V(lambda_m) delta_m^2
       + S_m c_m^2 W(lambda_m) delta_m^4].                        (8)

The diversity and curvature residuals are orthogonal in channel space: the first
uses b_m-p_m(a_m+b_m), while the second uses a_m+b_m. Consequently, on the
crossover path both noncentrality contributions add rather than cancel.

The corresponding vector expansion follows from the exact orthogonal channel
decomposition used in UNIFORM_CROSSOVER_THEORY.md. The diversity block is
delta_m times the normalized orthogonal-amplitude direction tensored with
P_perp{1,e}e', up to O(delta_m^2); the merged-amplitude block is c_m delta_m^2
times its normalized channel direction tensored with P_perp{1,e,e'}e'', up to
O(delta_m^3). The channel inner product is zero. Under the displayed local
scalings, multiplying these remainder vectors by sqrt(m) sends them to zero.
This argument additionally requires convergence of the two normalized channel
directions whenever their norms do not vanish.

For B_m/S_m asymptotic to m^(-a):

- a<1/2 and delta_m=kappa m^(-(1-a)/2): only the diversity normal direction
  survives in (2);
- a>1/2 and delta_m=kappa m^(-1/4): only the curvature direction survives;
- a=1/2 and delta_m=kappa m^(-1/4): both survive and (8) is their squared norm.

This supplies a non-degenerate power curve on every branch of the phase diagram,
not only the zero/infinity dichotomy.

## Scope and originality boundary

The smooth-manifold residual limit and noncentral chi-square distribution are
covered by general goodness-of-fit theory, including Shapiro, Xie and Zhang
(2019/2021): https://arxiv.org/abs/1909.05229. They are not P5 innovations.

The candidate contribution is the explicit normal-distance formula (8), its
uniform validity as amplitude diversity vanishes with the rate gap, and the
resulting change of local power path. Sharp minimax constants are still not
proved: (6) describes the profile residual test, while the earlier Pinsker bound
does not establish its optimality at every finite lambda_nc.

The signed/nonnegative constraint boundary, fixed-projection retention condition,
and iid Gaussian restrictions remain as stated in the preceding theory notes.

# Rate-optimal detection boundary under joint rate and amplitude degeneracy

Date: 2026-09-10. This note closes the necessary and sufficient detection scales
for the ideal union-residual test. It gives rate optimality up to constants, not
the exact minimax constant or local limiting power curve.

## Experiment and distance

Observe m independent C by n matrices

    Y_j = M_m + E_j,   vec(E_j) ~ N(0, sigma_m^2 I),  j=1,...,m.

The one-rate null M0 contains channel-specific signed amplitudes and baselines,
a shared rate k in a fixed compact interval K, and common unknown sigma. Let

    D_m = inf_{N in M0} ||M_m-N||_F,    rho_m=sqrt(m) D_m/sigma_m.

The infimum is attained: for each k, amplitudes and baselines are orthogonal
least-squares projections, and the profiled residual is continuous on compact K.

## Proposition 1: nonasymptotic obstruction

For every test phi with uniform null size at most alpha,

    E_Mm phi <= min{1, alpha + rho_m/2}.                         (1)

Proof. Let N_m be a closest null mean and compare m Gaussian observations with
means M_m and N_m and common variance sigma_m^2. Their KL divergence is
m D_m^2/(2 sigma_m^2)=rho_m^2/2. Pinsker gives total variation at most rho_m/2.
Since E_Nm phi<=alpha, (1) follows. This argument even grants the test knowledge
of sigma; hence the obstruction also applies when sigma is unknown.

Consequently rho_m -> 0 forces power to alpha+o(1). More quantitatively, no
level-alpha test can guarantee power 1-beta at this alternative when
rho_m < 2(1-beta-alpha).

## Proposition 2: finite-sample achievable bound

Let s^2 be the pooled within-repeat variance, nu=C n(m-1), d=C(n-2), and

    R_m=dist(Ybar,M0).

The ideal union-residual test rejects when

    m R_m^2/(d s^2) > F_{d,nu;1-alpha}.                          (2)

Its size is at most alpha uniformly over M0 and sigma>0. For any split
beta_1+beta_2=beta, its power is at least 1-beta whenever

    rho_m > T_m = sqrt(chi2_{C(n-1);1-beta_1})
      + sqrt[d F_{d,nu;1-alpha} chi2_{nu;1-beta_2}/nu].          (3)

Proof. At the true null rate, the numerator is dominated by a central chi-square
with d degrees of freedom and is independent of s^2, proving size. Distance to
a closed set is 1-Lipschitz. After projecting out channel baselines, the mean
noise norm is bounded by sigma sqrt(chi2_{C(n-1);1-beta_1}/m) with probability
1-beta_1. Independently, s/sigma is bounded by
sqrt(chi2_{nu;1-beta_2}/nu) with probability 1-beta_2. On their intersection,
(3) makes (2) hold. The union bound completes the claim.

For fixed C,n,alpha,beta, T_m converges to a finite positive constant as
m grows. Propositions 1 and 2 therefore match in the scale rho_m: rho_m -> 0 is
undetectable and rho_m -> infinity is consistently detectable. The constants
do not match and (3) is deliberately conservative.

For a formal minimax reading, let A_m(u) be any declared alternative class on
which rho_m<=u. Equation (1) bounds the supremum power at each member and hence
precludes uniform power 1-beta when the class contains a member with
u<2(1-beta-alpha). Conversely, on a class with inf rho_m>T_m, (3) gives uniform
power at least 1-beta. The phase result below supplies explicit parameter
classes for these two distance conditions. It does not solve minimax testing
over unrestricted two-rate means outside the compact crossover assumptions.

## Theorem: joint-degeneracy phase diagram

Adopt all compactness and interior assumptions of UNIFORM_CROSSOVER_THEORY.md.
Let S_m be bounded above and away from zero, p be bounded away from zero and one,
lambda remain in its declared compact interior, r_m=B_m/S_m be bounded, and
delta_m -> 0. Then uniformly over such sequences,

    rho_m^2 = (m/sigma_m^2)
      [B_m V(lambda_m) delta_m^2
       + S_m c_m^2 W(lambda_m) delta_m^4] [1+O(|delta_m|)].       (4)

Thus the bracket in (4), multiplied by m/sigma_m^2, is the detection boundary
up to constants. This conclusion combines the model-specific profiled distance
expansion with Propositions 1 and 2; the generic Gaussian propositions alone do
not produce the two-regime boundary.

For a transparent polynomial phase diagram, additionally keep sigma_m and S_m
of constant order and set r_m asymptotic to m^{-a}, a>=0. A critical separation
has delta_m asymptotic to m^{-tau(a)}, where

    tau(a) = (1-a)/2,  0<=a<1/2;
             1/4,      a>=1/2.                                  (5)

At a<1/2, the diversity term supplies the boundary; at a>1/2, the curvature term
does; at a=1/2 both contribute. In particular, fixed nonzero diversity yields
the m^{-1/2} separation scale, while proportional amplitudes yield m^{-1/4}.

Equation (5) is an order statement. Its constants depend on V,W,S,p,sigma and
the desired errors. It does not assert that delta alone identifies order when a
mode amplitude vanishes or when the design degenerates.

## Fixed projection

Let Q be fixed independently of the evaluation mean, Q^T Q=I and Q^T 1=0.
Retain the complete projected nonlinear null curve. If, on the declared local
alternative family, D_{Q,m}>=eta D_m for a constant eta>0, the projected test
has the same exact size argument with d=C(q-1), and its sufficient condition is
rho_m>T_m(q)/eta. It therefore preserves (4)-(5). A vanishing eta can change or
destroy the detection rate; a finite-grid numerical eta is not a theorem.

## What is and is not closed

Closed here: a nonasymptotic lower obstruction, a finite-sample sufficient
upper bound, and matching dependence on the scalar distance rho_m; substitution
of the uniform crossover yields the two detection exponents and their joint
transition.

Not closed: sharp minimax constants; the limiting experiment when rho_m tends
to a finite constant; a computable delta_0 and remainder constant; continuum
certification of eta; machine-directed rounding; order-two certification after
one-rate rejection; colored or heteroscedastic noise.

The main originality risk is conceptual genericity. Testing by Gaussian distance
and T-optimal discrimination are classical. Any novelty claim must attach to the
uniform, nuisance-profiled two-term distance in (4) and the resulting joint phase
diagram, not to Pinsker, the F test, or experimental design in general.

All distance statements above use the signed-amplitude null. They transfer to a
nonnegative-amplitude null only under the inactive-constraint condition in
UNIFORM_CROSSOVER_THEORY.md, which ensures that the closest local signed fit is
feasible. Without that condition, the lower-bound comparison mean need not lie
in the smaller nonnegative null, so the same phase formula must not be asserted.

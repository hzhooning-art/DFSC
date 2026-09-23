# Uniform diversity crossover for profiled shared real exponentials

2026-09-10. Research theorem with a proof below; constants are existential,
not a computable finite-gap certificate. Originality remains unestablished.

## Statement

Fix n>=4 distinct nonnegative times and K=[l,u], 0<l<u. Let lambda range in
a compact subset of the interior of K, p in [epsilon,1-epsilon], epsilon>0,
and r=B/S in [0,R], R finite. S=||a+b||^2>0, p=(a+b)^T b/S,
B=||b-p(a+b)||^2. Write e=e(lambda), e'=d e/dlambda, e''=d²e/dlambda²,

V(lambda)=||P_perp{1,e}e'||²,
W(lambda)=||P_perp{1,e,e'}e''||², c=p(1-p)/2.

Let D be the global Frobenius distance of
M=a e(lambda)^T+b e(lambda+delta)^T+baseline*1^T
to one shared rate in K with arbitrary real channel amplitudes and baselines.
There exist delta0>0 and C<infinity, depending only on the fixed compact
parameter domains and the design, such that for |delta|<=delta0,

 |D²/S - [r V delta² + c² W delta⁴]| <= C [r |delta|³ + |delta|⁵].       (1)

V,W are uniformly positive on the lambda domain. Therefore, uniformly over
the whole r interval, including any r=r(delta) tending to zero,

 D² = [B V delta² + S c² W delta⁴] [1+O(|delta|)].                     (2)

The relative statement uses p bounded away from 0,1. Equation (1) is an
absolute expansion and need not imply (2) at degeneracies of its leading term.

## Proof

1. Exact profiling gives D²/S=min_k J(k;r,delta), where
J=||P_k f||²+r||P_k g||², g=e(lambda+delta)-e(lambda), f=e(lambda)+p g,
P_k=P_perp{1,e(k)}. This is the earlier exact channel-orthogonal decomposition.

2. At delta=0, J=||P_k e(lambda)||², independently of r,p. Its only zero on K
is k=lambda. Indeed, a nonzero combination of 1,exp(-lambda*t),exp(-k*t)
has at most two distinct zeros when k!=lambda. Three distinct observation
times rule out an affine relation between these exponentials. On compact
sets away from k=lambda the objective has a positive uniform separation.
The objective at k=lambda tends uniformly to zero as delta tends to zero,
so every global minimizer must approach lambda uniformly.

3. At (k,delta)=(lambda,0), J_kk=2V(lambda)>0. The columns 1,e,e' are
independent on at least three distinct times. Continuity, compactness and
the implicit function theorem produce a unique smooth stationary branch
k*(r,delta)=lambda+p delta+O(delta²), uniformly on the compact parameter set.
The separation in step 2 and positive curvature in a common neighborhood
make this branch the unique GLOBAL minimizer for sufficiently small delta.
One can extend r slightly below zero locally to apply the smooth theorem
at r=0; the global objective here is only asserted for r>=0.

4. Set H(r,delta)=J(k*(r,delta);r,delta). By the envelope identity,
dH/dr=||P_{k*(r,delta)}g||².
Uniformly, g=delta e'+O(delta²) and P_{k*}=P_lambda+O(delta), hence
dH/dr=delta² V+O(|delta|³). Integrating from 0 to r yields
H(r,delta)-H(0,delta)=r delta² V+O(r |delta|³).
This step retains the factor r in the remainder; separately derived fixed-r
and r=0 expansions alone would not establish a uniform crossover.

5. For r=0, f=e+p delta e'+(p/2)delta² e''+O(delta³).
The fitted rate is lambda+p delta+q delta²+O(delta³); fitted amplitude is
1+A delta²+O(delta³), fitted baseline is L delta²+O(delta³).
Their second-order residual is delta²[c e''-q e'-A e-L1]. Minimizing its
norm gives c P_perp{1,e,e'}e''. Smoothness of the minimizing branch gives
H(0,delta)=c² W delta⁴+O(|delta|⁵), uniformly in lambda,p.
Combine with step 4 to obtain (1). The functions 1,e,e',e'' form an
extended exponential Chebyshev system, giving W>0 on four distinct times.
Compactness and epsilon>0 bound V,c²W below, yielding (2).

## Scope and interpretation

The two leading terms balance at r = (c²W/V) delta². Thus diversity strength
sqrt(B/S) has a design- and mixture-dependent crossover proportional to |delta|.
For r(delta) proportional to |delta|^(2gamma), D scales as
sqrt(S)*|delta|^(min(1+gamma,2)); at gamma=1 both terms contribute.
These are geometry exponents, NOT an achieved minimax detection rate.
Gaussian pairwise KL is D²/(2sigma²); it gives necessary testing obstructions,
not a matching achievable algorithm or a finite-gap uniform level guarantee.

If all merged amplitudes s_i=a_i+b_i are uniformly positive relative to sqrt(S),
the locally fitted amplitudes remain positive for small delta under bounded r.
Then the signed global minimizer is feasible for the nonnegative null, so the
same result holds there. Without that inactive-constraint condition do not
transfer the signed-null formula to the nonnegative-null distance.

Exclusions: n=3 permits W=0; p=0 or 1 can eliminate a component at r=0;
lambda at the constrained rate boundary can block the first-order fitted shift;
unbounded B/S or merged amplitude zero escapes the compact theorem. Very small
lambda or a collapsing time design may destroy uniform constants. The theorem
does not supply a numerical delta0 or C. Monte Carlo/grid agreement cannot do so.

Related primary work already establishes amplitude-incoherence benefits in
multi-snapshot Fourier spectral estimation:
https://arxiv.org/abs/2202.11189
https://www.sciencedirect.com/science/article/pii/S1063520324000502
Our candidate distinction is this nuisance-profiled real-decay crossover with
free baselines and a diversity-weighted remainder, not the general diversity
principle. A detailed theorem-to-theorem comparison is still needed.

## Frozen numerical checks

24 times on [0,8], K=[.2,.8], lambda=.3,.45,.6, p=.2,.5,.8,
delta=.0002,.0005,.001,.002,.005,.01,.02; S=1.
Paths r=0,.01,delta,delta³, and z*(c²W/V)*delta² with z=.1,1,10.
441 cases. Screen stationary points over the full rate interval, refine roots,
and compare endpoints. This is numerical global screening, not certification.
Record every error. Predeclared diagnostic: all delta<=.002 cases have leading
squared-distance relative error <=5%; no claimed finite-gap theorem from this.
Boundary checks: n=3 (curvature vanishes), absent mode p=0,r=0 (distance zero),
and lambda at K's upper endpoint with positive gap (blocked rate shift).
Independently verify selected crossover cases with high-precision arithmetic.

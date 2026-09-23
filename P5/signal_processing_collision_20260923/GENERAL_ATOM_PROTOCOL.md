# Frozen smooth-atom extension audit

Date frozen: 2026-09-11, before computing outcomes.

The generic distance theorem is checked on two non-exponential sampled atom
curves: Gaussian scale atoms `exp(-theta*t^2)` and rational decay atoms
`1/(1+theta*t)`.  Each design uses 24 fixed times, an intercept nuisance,
three channel-amplitude proportions `p` in `{0.2,0.5,0.8}`, gaps in
`{0.002,0.005,0.01,0.02}`, and diversity ratios `B/S` on the proportional,
crossover, and fixed-diversity paths.

For each cell, the squared distance to the one-atom model is globally profiled
over a dense parameter grid followed by bounded scalar refinement.  The audit
compares it with `r*V*delta^2 + c^2*W*delta^4`, where `V` and `W` are computed
from the analytic first and second atom derivatives.

Frozen gates:

1. projected derivative coefficients `V` and `W` are positive in every design;
2. every profiled minimizer is interior;
3. at gaps at most 0.005, the maximum relative leading-law error is at most 5%;
4. for each family/path/proportion combination, relative error at gap 0.002 is
   no larger than at gap 0.02.

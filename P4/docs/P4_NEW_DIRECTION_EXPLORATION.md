# P4 New-Direction Exploration

## Current status

The following P4 candidates have been stopped as independent-paper routes:

- memory-kernel identification;
- parameter-conditioned forward-family learning;
- EAFI structural-mismatch detection;
- pointwise error-weighted active observation design.

Their negative results remain in the experiment records. This document starts a
fresh route audit rather than extending those experiments.

## Boundary conditions from P1--P3

- P1 already owns the differentiable Mittag-Leffler primitive and its batched
  GPU computational graph.
- P2 owns value/gradient error control, evaluator dispatch, and reliability
  auditing.
- P3 owns structured-plus-neural residual composition and selective hybrid
  inference.

A viable P4 must therefore introduce a new scientific decision variable or
algorithmic object, not another layer variant, another error threshold, or
another residual architecture.

## Candidate portfolio

### Candidate A: task-aware spectral budget allocation

**Question:** Given a batch containing different propagation times, parameters,
and gradient requirements, how should a fixed compute budget be allocated over
spectral modes, series terms, or Krylov dimensions?

**Distinct object:** a batch-level resource allocation policy, rather than an
evaluator error certificate. It would optimize accuracy-throughput trade-offs
under a global budget and could choose different budgets for forward-only and
gradient-bearing samples.

**Potential evidence:** Pareto curves of value error, gradient error, wall time,
GPU memory, and energy proxy; comparison with uniform budget and P2's fixed
reliability policy.

**Risk:** It may be judged as an engineering extension of P2 unless the policy
has a principled constrained optimization formulation and improves a clearly
defined scientific workload metric.

### Candidate B: differentiable spectral operator compression

**Question:** Can a Mittag-Leffler propagator be compressed into a task-aware,
gradient-preserving reduced representation whose error is controlled across a
parameter region rather than at one evaluation point?

**Distinct object:** a reduced operator representation and its certification,
not a new propagator or a standard neural surrogate.

**Potential evidence:** compression ratio, forward/gradient error, memory, GPU
throughput, and transfer across unseen parameter values; comparisons with low
rank, Chebyshev, and generic neural surrogates.

**Risk:** Generic matrix-function compression and neural operator distillation
are mature areas. The route is only interesting if the fractional spectral
structure yields a measurable advantage and the gradient guarantee is central.

### Candidate C: fractional state realization for long-horizon query serving

**Question:** Can the known Mittag-Leffler evolution be represented by a compact
latent state that supports many irregular time queries and backpropagation with
bounded drift, without history replay?

**Distinct object:** a query-serving state realization and its long-horizon
drift analysis, rather than one-step propagation or neural residual learning.

**Potential evidence:** irregular-query accuracy, horizon scaling, memory use,
gradient drift, and amortized cost per query; comparisons with history-based
L1 convolution, recurrent state models, and direct MLSL evaluation.

**Risk:** It may overlap with existing state-space approximations for fractional
systems. The route requires a clearly fractional-specific construction and a
strong long-horizon advantage.

### Candidate D: fractional operator preconditioning for inverse/physics losses

**Question:** Can the known fractional propagator be used to precondition the
  optimization geometry of a neural or differentiable inverse problem, reducing
  stiffness and improving convergence without adding a residual model?

**Distinct object:** an optimization preconditioner acting on parameter or
functional gradients, not a forward primitive or an evaluator controller.

**Potential evidence:** condition-number proxies, iterations to target loss,
  gradient noise, sensitivity to fractional order, and matched-budget training
  curves against Adam, natural-gradient approximations, and unpreconditioned
  baselines.

**Risk:** The contribution may look like a domain-specific optimizer trick
  unless it has a general derivation and transfers across at least two inverse
  task families.

## Preliminary ranking

| Candidate | Independence from P1--P3 | Scientific upside | Feasibility | Main risk | Priority |
|---|---:|---:|---:|---|---:|
| A. Budget allocation | medium | medium | high | looks like P2 engineering | 2 |
| B. Spectral compression | high | high | medium | generic compression overlap | 1 |
| C. Long-horizon state realization | high | high | medium-low | existing fractional state-space work | 3 |
| D. Optimization preconditioning | medium-high | medium-high | medium | optimizer-specific contribution | 4 |

## Recommended next route

Candidate B, **differentiable spectral operator compression**, is the strongest
new hypothesis because its primary object is a reusable reduced representation
of a family of propagators. It can be tested without the unstable forcing
primitive and without P3's residual model.

The first feasibility test should not train a neural surrogate. It should ask
whether a fractional-structure-aware reduced representation can achieve a
better forward/gradient-error versus storage/throughput Pareto curve than:

1. uniform spectral truncation;
2. generic low-rank/SVD compression;
3. Chebyshev or polynomial matrix-function approximation;
4. an uncompressed MLSL reference.

## Hard stop for Candidate B

Stop immediately if any of the following occurs:

- compression gives no Pareto improvement over generic low rank;
- gradient error grows materially faster than value error;
- the advantage only appears for one hand-picked parameter point;
- the reduced representation cannot support batched GPU evaluation;
- the contribution requires a neural surrogate to show any benefit.

The scalar Chebyshev probe is recorded in
`P4_SPECTRAL_COMPRESSION_FEASIBILITY.md`, and the final matrix-scale probe is
recorded in `P4_MATRIX_COMPRESSION_DECISION.md`. Both failed the Pareto gate:
direct MLSL was faster and more accurate, while reduced spectral ranks caused
large value and gradient errors. Candidate B is therefore closed as an
independent-paper route. Candidates A and D remain secondary because they are
more likely to be interpreted as P2 extensions. Candidate C is now the only
remaining high-independence route for a bounded feasibility test; its detailed
plan is in `P4_FRESH_ROUTE_STATE_REALIZATION.md`.

## Positioning rule

No new P4 paper title or contribution claim should be finalized before the first
candidate passes its hard stop and a focused literature check confirms that the
specific fractional-gradient compression problem is not already solved.

# P4 Fresh Route: Long-Horizon Fractional State Realization

## Why this route is being considered

The following candidates have been stopped after explicit tests:

- inverse identification under structural mismatch;
- pointwise error-aware active observation design;
- scalar Chebyshev compression;
- matrix spectral truncation.

The remaining P4 candidate is a different computational object: a compact
state realization for repeated, irregular-time fractional propagation.

## Research question

Can a fractional propagator be represented by a compact differentiable state
that supports many irregular time queries and long-horizon rollout while
preserving forward accuracy and parameter-gradient stability better than
history-based accumulation or repeated direct evaluation?

The question is not whether a fractional equation can be approximated by a
sum of exponentials in general. That literature is established. The possible
DFSC-specific contribution would have to be a differentiable, batched, GPU
state realization with an explicit forward/gradient drift contract for
Mittag-Leffler propagators.

## Separation from P1--P3 and P5

- **P1:** evaluates a differentiable propagator at a requested point. P4 would
  maintain a reusable state across many irregular requests.
- **P2:** audits the value and gradient of one evaluator. P4 would study error
  accumulation and drift over a query sequence.
- **P3:** composes a structured backbone with a neural residual. P4 would not
  use a learned residual or routing policy.
- **P5:** transports an operator between representations. P4 would keep one
  representation and study temporal state evolution.

## Candidate state construction

The first implementation should use a positive exponential-sum realization,

\[
  h_j'(t)=-r_j h_j(t)+b_j u(t),\qquad
  y(t)=\sum_j c_j h_j(t),
\]

with analytic interval updates for irregular time gaps. The rates and weights
must be fitted or selected on a declared alpha/time envelope, then frozen
before test evaluation. No neural network should be introduced in the first
gate.

## Minimum experiment

1. Fit state realizations for a small alpha envelope and a fixed time horizon.
2. Query the same trajectory at irregular times and compare with direct MLSL.
3. Measure forward error and alpha/rate gradient error after 10, 100, and 1000
   queries.
4. Compare direct MLSL, history-based L1 convolution, and the state realization
   at matched tolerances.
5. Repeat with different query orderings and batch sizes on CPU and GPU.

## Evidence required for an independent paper

- lower amortized cost per query than direct evaluation or history replay;
- no long-horizon gradient drift larger than the forward error budget;
- stable behavior under irregular query gaps;
- at least two alpha/time envelopes or two operator families;
- a failure regime where the state budget is insufficient and the method
  reports rejection or refinement rather than silently returning a result;
- comparison against an established exponential-sum/state-space baseline.

## Hard stop

Stop this P4 route if any of the following occurs:

- the state realization is only competitive after using a larger state than the
  direct representation;
- gradient drift grows faster than forward error with query count;
- the method only works for one fixed alpha and one fixed time window;
- it cannot beat history replay in amortized cost at matched accuracy;
- the differentiable state construction is not materially different from an
  existing fractional exponential-sum method.

If the route fails, P4 should be discontinued as an independent paper. The
remaining defensible project endpoint is then the P1--P3 DFSC series plus
software-level utilities, rather than an artificially extended five-paper
sequence.

## Immediate status

The first state-realization feasibility result is recorded in
`P4_STATE_REALIZATION_DECISION.md`. It failed the accuracy, gradient, and
throughput gates, so this route is closed as an independent P4 paper direction.

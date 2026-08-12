# P4: Differentiable Primitive Reliability Protocol

**Route status:** P4 has converged to a protocol and software artifact for
auditing differentiable scientific-computing primitives in scientific machine
learning. The manuscript claim is deliberately narrower than a universal
solver claim: it specifies what must be measured before a primitive is treated
as reliable and reusable on a declared domain.

The protocol is instantiated with MLSL as the motivating fractional case study,
and with matrix-exponential, linear RK4, and Logistic RK4 backends as
cross-primitive controls. The current evidence supports a standalone P4 paper
about reproducible evaluation and module reuse, subject to the scope limits in
`docs/GENERAL_DIFFERENTIABLE_PRIMITIVE_PROTOCOL.md`.
The first feasibility route uses a positive normalized mixture of exponential
kernels,

\[
k(t)=\sum_{j=1}^M w_j r_j \exp(-r_j t),\qquad
w_j\geq 0,\quad \sum_j w_j=1,\quad r_j>0.
\]

This parameterization is causal and non-negative by construction. It is a
finite-dimensional approximation to a broader positive memory-kernel family,
not yet a general fractional solver or a claim of complete monotonicity for
every future extension.

## Paper assets

- `paper/dfsc_primitive_protocol_en.tex`: English manuscript draft.
- `paper/dfsc_primitive_protocol_zh.tex`: synchronized Chinese draft.
- `paper/build_paper_data.py`: rebuilds the auditable paper-data bundle.
- `paper/paper_data.json`: generated data bundle with source-file provenance.

The drafts distinguish direct generic-protocol experiments from prior MLSL
validation and retain the negative matrix-backend module-level result.

## Historical feasibility material

`experiments/feasibility_positive_mixture.py` fits a positive exponential mixture
to a tempered power-law target. It reports kernel fit, causal convolution error,
held-out long-time error, positivity, normalization, and gradient finiteness.

## Current P4 boundary

- P1 supplies the differentiable propagation primitive and software base.
- P2 supplies numerical reliability diagnostics for propagation and gradients.
- P3 studies when to activate a learned correction around a fixed structured
  operator.
- P4 maintains a reusable temporal state under irregular query requests.

The current result is sufficient for a protocol-focused draft, but not for a
claim of a complete fractional solver ecosystem. Remaining evidence gaps are
hosted CI, independent external reproduction, broader numerical domains, and
at least one public physical benchmark if the target venue requires application
evidence.

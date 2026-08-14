# P4 Integrated Workflow: Initial Evidence

## Scope

The simplified P4 route combines three questions into one workflow:

1. reliability evaluation of the differentiable fractional primitive;
2. batched differentiable alpha/lambda calibration;
3. reuse inside neural scientific-computing modules.

The research object is now a backend-independent reliability protocol for
scientific-computing primitives. MLSL is the first substantive backend case
study; the generic interface is tested separately with an exponential
propagator smoke backend.

The first benchmark and calibration results are recorded in the existing P4
reports. This document records the initial cross-module result.

## Calibration evidence

For 32 noisy, off-grid tasks, batched MLSL calibration achieved a mean joint
parameter error of `0.00919`, compared with `0.01056` for a 41x41 grid search.
The grid search was faster at this small scale (`0.244 s` versus `0.615 s`),
so the current evidence supports differentiability and competitive recovery,
not a universal speed claim.

## Cross-module evidence

The experiment trained three modules on 512 tasks and evaluated them on 256
tasks sampled from a wider alpha/rate domain. Results are mean +/- standard
deviation over three seeds.

| module | OOD RMSE | standard deviation | training time |
|---|---:|---:|---:|
| pure MLP | 0.02842 | 0.00236 | 0.548 s |
| MLSL encoder-decoder | **0.02020** | **0.00077** | 1.202 s |
| MLSL + residual adapter | 0.02037 | 0.00146 | 1.243 s |

All gradients were finite. The MLSL encoder-decoder improves the controlled
OOD prediction error, but costs more training time. The residual adapter adds
no clear benefit in this first test and must not be presented as a new neural
architecture.

The raw result is stored in
`P4/results/p4_module_reuse_feasibility.json`.

## Operator-style batch evidence

An additional 16-mode, 16-query operator task used 1024 training fields and
256 OOD test fields. Direct MLSL propagation achieved RMSE `2.35e-8` and finite
alpha/rate gradients under a batched loss. A pure MLP surrogate reached OOD
RMSE `0.0481` after training. The direct MLSL operator path therefore provides
an accurate differentiable reference that can be inserted into an operator-
style workflow; this comparison is not a new operator-learning architecture
claim.

The raw result is stored in
`P4/results/p4_operator_batch_reuse.json`.

## Physics-consistency reuse evidence

To test whether the primitive can serve as a differentiable physical constraint,
we trained the same observation encoder with either a data-only objective or a
joint objective containing an MLSL trajectory residual. The training domain
contained 512 tasks and the wider test domain contained 256 tasks; each result
is the mean +/- standard deviation over three random seeds.

| objective | OOD RMSE | parameter error (alpha / rate) | physics residual | training time |
|---|---:|---:|---:|---:|
| data only | **0.02402 +/- 0.00193** | 0.06354 / 0.12528 | 0.06472 +/- 0.00379 | 1.751 s |
| data + MLSL consistency | 0.04246 +/- 0.00074 | **0.06127 / 0.12335** | **0.02913 +/- 0.00319** | 1.663 s |

All gradients were finite. The consistency term substantially reduces the
physical residual and slightly improves parameter recovery, but worsens the
field prediction error under this fixed weight and sparse-observation setup.
This is evidence for a reusable differentiable audit/regularization role, not
evidence that adding the primitive always improves prediction accuracy. Weight
selection and task-dependent trade-offs remain open.

The raw result is stored in
`P4/results/p4_physics_consistency_reuse.json`.

### Matched-weight sensitivity

We then swept the physics weight over `{0, 0.05, 0.10, 0.25, 0.50, 1.0}`
under the same data, architecture, 600-step budget, and three seeds. The mean
OOD RMSE increased from `0.02206` at weight `0` to `0.04224` at weight `1.0`,
whereas the mean physics residual decreased from `0.06311` to `0.02730`.
Parameter errors improved slightly at intermediate weights, but no nonzero
weight improved both prediction RMSE and physical residual relative to the
data-only model. Under the predeclared rule of minimizing OOD RMSE subject to
non-increasing physical residual, the selected weight was `0.0`.

This sensitivity result is important evidence against an unconditional
physics-informed improvement claim. It supports exposing the MLSL term as an
explicit, user-controlled audit or regularization component whose weight must
be selected for the task. The raw sweep is stored in
`P4/results/p4_physics_weight_sweep.json`.

## Interpretation

The reusable protocol is defined in
`P4/docs/P4_PRIMITIVE_RELIABILITY_PROTOCOL.md`, and the machine-readable gate
summary is written to `P4/results/p4_primitive_reliability_audit.json`.

The integrated workflow has a plausible practical story:

- reliability tests establish when the primitive can be trusted;
- calibration demonstrates direct differentiable use;
- the encoder-decoder and operator tests demonstrate reuse in neural modules;
- the physics-consistency test demonstrates a measurable value--physics trade-off.

Taken together, the evidence supports a practical integrated-workflow claim:
the MLSL primitive can be batched, differentiated with respect to model
parameters, used as an operator reference, and inserted as a physical audit
term. It does not support a claim of a new PINN architecture, universal OOD
improvement, or guaranteed speedup. The matched-weight sweep has now completed
the current evidence gate; the independent-paper claim should remain limited
to a reproducible primitive reliability and reuse protocol. A stronger
algorithmic claim would require a new task where a selected nonzero weight
improves a predeclared primary metric without sacrificing the audit metric.

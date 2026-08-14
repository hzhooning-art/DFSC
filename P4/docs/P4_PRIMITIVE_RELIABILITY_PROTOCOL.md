# Differentiable Scientific-Computing Primitive Reliability Protocol

> This document is backend-independent. The current MLSL experiments are one
> instantiation, not the scope definition of the protocol.

## Purpose

This protocol defines how DFSC should report a differentiable scientific-
computing primitive in scientific machine learning. It separates
function-value fidelity,
parameter-gradient validity, differentiable calibration, and module reuse.
It is a usage and audit standard, not a claim of a universal fractional solver
or a new neural architecture.

## Required declaration

Every experiment must report the operator family and parameter domain, basis and
retained modes, evaluator terms, dtype, device, batch shape, whether the path
is a known-propagator evaluation or a learned surrogate, primary value and
gradient metrics, random seeds, train/test domains, and timing protocol.

The current validated path is the real, stable Mittag--Leffler propagation
family implemented by the MLSL evaluator. Claims outside that family require a
separate validation record.

## Reliability gates

### A. Function values

Compare MLSL with a high-precision reference on off-grid times and a held-out
parameter grid. Report absolute and relative error, failure counts, and
long-horizon error separately. Fix the pass threshold before the run.

### B. Parameter gradients

Check finite gradients for every trainable parameter and compare autograd with
finite differences or a high-precision directional derivative. Report the
maximum relative directional-gradient error and failed probes. Adaptive
evaluation decisions must be detached from the optimization graph while the
selected computation remains differentiable.

### C. Parameter calibration

Use noisy, off-grid observations and compare batched gradient calibration with a
stated non-gradient baseline. Report alpha/rate errors, joint error, loss,
runtime, and seed variability. A speed claim requires matched grid coverage,
precision, device, and objective.

### D. Module reuse

Test both an encoder-decoder or residual neural module and a batched
operator-style loss or physical-consistency objective. Report OOD prediction
error, parameter recovery, physical residual when relevant, training time,
memory/device information, and gradient finiteness. Improvements from a
separate architecture must not be attributed to the primitive.

### E. Trade-off audit

When MLSL is used as a physical loss or regularizer, sweep its weight including
zero under a matched training budget. Declare the selection rule before seeing
the results. Report the Pareto trade-off among prediction error, parameter
error, and physical residual. Do not claim universal improvement when the sweep
shows a trade-off.

## Current status

The current P4 evidence passes the batched/autograd/reuse checks and includes
three-seed calibration and module evaluations. The physical-loss sweep shows a
reproducible trade-off: increasing the physics weight lowers the MLSL residual
but raises OOD RMSE in the tested sparse-observation setting. The recommended
default is therefore an explicit, user-controlled audit or regularization term
with task-specific weight selection.

## Prohibited conclusions

The current evidence does not justify claiming that DFSC replaces general
fractional solvers, always improves neural prediction accuracy, is faster than
every classical or learned baseline, introduces a new PINN/operator-learning
architecture, or is validated for variable-order, distributed-order,
complex-geometry, or unstructured-grid problems.

# P4 New Route Plan: Error-Aware Active Experiment Design

## Immediate decision

The previous P4 route, **Error-Aware Fractional Operator Identification under
Structural Mismatch**, is frozen. In particular, the automatic near-mismatch
detector is closed after the discrepancy-ladder failure documented in
`P4_EAFI_DISCREPANCY_LADDER.md`.

The new P4 candidate is:

> **Error-aware active experiment design for differentiable fractional
> propagators**: choose a limited set of observation times or channels that
> maximizes parameter identifiability while accounting for numerical value and
> gradient error.

This is a planning hypothesis, not yet a novelty claim. A literature check and
feasibility gate are required before implementation is expanded.

## Why this is a distinct scientific object

| Paper | Primary object | What P4 will not repeat |
|---|---|---|
| P1 | Differentiable Mittag-Leffler spectral primitive | P4 will not introduce another layer or claim a new propagator |
| P2 | Value/gradient evaluator reliability | P4 will use the error report as a design constraint, not as its main algorithm |
| P3 | Structured backbone plus neural residual routing | P4 will not train or route a hybrid neural forward model |
| New P4 | Observation/sensor design for fractional parameter identifiability | P4 changes the decision variable from model parameters to measurement design |
| P5 | Representation transport and commutation diagnostics | P4 will use a fixed representation and will not study mesh/basis transfer |

## Core research question

Given a measurement budget `m`, observation domain `T`, noise model, and a
numerical evaluator with estimated value/gradient error, can a differentiable
design rule select observations that improve recovery of fractional order and
propagation parameters compared with uniform or random sampling?

The design objective should combine information and numerical reliability, for
example

\[
  J(S)=\log\det(F_S+\lambda I)
       -\rho\,E_S^{\rm value}
       -\eta\,E_S^{\rm gradient},
\]

where `S` is the selected observation set, `F_S` is a Fisher-information or
local sensitivity matrix, and the last two terms penalize evaluator uncertainty.
The exact objective must be frozen after the first feasibility study; it must
not be selected after seeing the final test results.

## Expected independent contribution

The paper would contribute an **error-aware design criterion and selection
procedure**, not merely apply a standard optimal-design algorithm to a
fractional equation. Its central test is whether accounting for differentiable
propagator error changes the selected measurements and improves out-of-sample
parameter coverage per observation.

The minimum publishable claim requires all of the following:

1. The error-aware design must beat uniform sampling and at least one standard
   sensitivity/Fisher design on parameter recovery or coverage per measurement.
2. The advantage must remain after the design is frozen and evaluated on new
   noise seeds and off-grid parameters.
3. The gain must persist when the evaluator budget is changed; otherwise it is
   only a tuning artifact.
4. The result must include a failure regime, such as weakly identifiable
   orders, high noise, or a short observation horizon.

## Feasibility stages

### Stage A: local design sanity check

- Scalar `E_alpha(-lambda t^alpha)` family.
- Compare uniform, random, Fisher/D-optimal, and error-aware selection.
- Measure condition number, log-determinant, gradient finiteness, and runtime.
- Use separate design and evaluation seeds.

The first Stage-A result is recorded in `P4_ACTIVE_DESIGN_FEASIBILITY.md`.
The initial pointwise error-weighted D-optimal rule failed against standard
D-optimal selection and is frozen as a rejected candidate. Only one bounded
robust/Pareto redesign remains before the whole P4 direction is stopped.

### Stage B: finite-budget inverse recovery

- Use sparse observations with noise levels and off-grid parameters.
- Fit `alpha` and `lambda` using only selected observations.
- Report parameter error, empirical interval coverage, retained-area/volume,
  and observations required to reach a target error.

### Stage C: evaluator-budget stress test

- Repeat with low, medium, and high evaluator terms.
- Verify that the design reacts to numerical error rather than silently
  selecting high-error time regions.
- Compare CPU/GPU overhead and batched selection cost.

### Stage D: broader family and external validation

- Add at least one additional known-propagator fractional family or public
  time-series task only after Stages A--C pass.
- Freeze the objective and selection hyperparameters before this stage.

## Baselines

- Uniform time sampling.
- Random sampling with matched budget.
- Classical D-optimal/Fisher-information selection using the high-accuracy
  evaluator but no error penalty.
- A sensitivity-only greedy selector.
- Oracle high-accuracy design, reported only as an upper reference.

## Exit conditions

Stop the route if any of the following occurs:

- error-aware selection is indistinguishable from uniform sampling across the
  first three stages;
- it improves local information but not held-out parameter recovery or
  coverage;
- the claimed gain disappears when evaluator terms or noise seeds change;
- the method requires a problem-specific objective retuned for every test case;
- the method's value is only lower wall-clock time with no inferential benefit.

If Stage B passes but Stage D does not, retain P4 as a methodological paper on
controlled fractional experiment design. If Stage B fails, stop P4 as an
independent paper and preserve the implementation as an optional DFSC module.

## Relation to future P5

P5 remains reserved for **certified representation transport**: forward and
gradient commutation defects across bases, grids, or resolutions. P4 must not
expand into mesh transfer, and P5 must not absorb measurement design. This keeps
the paper boundary based on the scientific decision variable: observations for
P4, representations for P5.

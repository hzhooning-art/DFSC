# General Protocol for Differentiable Scientific-Computing Primitives

## Research object

The target of this work is not a particular Mittag--Leffler layer. The target
is a backend-independent protocol for deciding whether a scientific-computing
primitive is reliable and reusable inside scientific machine learning.

MLSL is the first case study because it has an explicit differentiable
propagator and a nontrivial parameter-calibration problem. It is not the
definition of the protocol and should not be used to claim that all primitives
have the same numerical behavior.

## Primitive contract

Each backend must expose one batched differentiable map

\[
    y = P(x;\theta),
\]

with an explicit input domain, parameter domain, output contract, dtype policy,
device policy, and reference evaluator. The primitive may be an integrator,
matrix-function action, spectral layer, PDE stencil, differentiable solver
step, or structured neural-physics block.

## Four normalized dimensions

### 1. Numerical fidelity

Measure value error against a declared reference on in-domain, off-grid, edge,
and long-horizon cases. Report absolute error, relative error, finite-value
failure rate, and error as a function of computational budget.

### 2. Differentiability

Measure finite-gradient rate, directional-gradient error against finite
differences or a high-precision derivative, sensitivity across parameters, and
whether adaptive control decisions preserve a valid backward graph.

### 3. Parameter calibration

Use noisy observations and a declared inverse objective. Report parameter error,
joint error, convergence failure rate, seed mean/std, runtime, and comparison
with a matched non-gradient baseline. A differentiable primitive is not
automatically a good estimator; calibration is a separate gate.

### 4. Module reuse

Test the same primitive in at least two compositions: a neural encoder/decoder
or residual module, and a batched operator or physics-consistency objective.
Report OOD error, physical residual where meaningful, memory, device locality,
training/inference cost, gradient finiteness, and interface changes required.

## Standard experiment matrix

Every backend should be evaluated on the same matrix where its mathematics
permits it:

| axis | minimum comparison |
|---|---|
| value | reference evaluator or high-precision computation |
| gradient | finite-difference/directional derivative |
| calibration | gradient optimization vs matched grid/derivative-free baseline |
| reuse | pure neural module vs primitive-composed module |
| scale | batch size, parameter count, horizon, device, dtype |
| robustness | noise, edge parameters, OOD parameters, random seeds |

## Pass/fail semantics

The protocol reports gate status rather than a single universal score. A backend
passes only the dimensions it actually measures. “Pass with scope limits” means
the tested contract is reliable on the declared domain; it does not extend the
claim to unsupported parameter regimes, geometries, orders, or backends.

## Current MLSL instantiation

The existing MLSL evidence instantiates this generic protocol as follows:

- value and gradient checks: stable real Mittag--Leffler propagation family;
- calibration: noisy, off-grid alpha/rate recovery;
- module reuse: encoder-decoder, batched operator reference, and physics audit;
- execution: PyTorch CPU/GPU path with batched tensors;
- limitation: no claim yet for general fractional solvers, variable-order
  operators, distributed-order operators, or non-structured geometries.

The generic implementation is in `P4/primitive_protocol.py`. The current MLSL
results remain a case study and should be reported as one backend row in a
future cross-primitive benchmark, not as the protocol itself. A first
cross-primitive audit now includes MLSL, an exponential propagator interface
smoke backend, and a substantive 2x2 matrix-exponential-action backend. The
matrix backend additionally passes five-seed OOD and long-horizon checks over
`0.02--12` time units, with mean long-horizon maximum absolute error
`4.27e-15`, mean directional-gradient relative error `3.79e-11`, and mean
calibration parameter L1 error `3.11e-4`. The cross-backend summary is stored
in `P4/results/p4_cross_primitive_audit.json`.

A third substantive backend, a fixed-step differentiable RK4 step for the
linear ODE `y'=Ay`, also passes the basic gates. Its one-step maximum absolute
error is `1.25e-5` on the declared step-size range, directional-gradient
relative error is `5.06e-11`, parameter calibration L1 error is `6.52e-4`, and
100 steps to `t=5` give absolute state error `1.60e-9`. These values are not
pooled with MLSL or matrix-exponential errors; they illustrate why a general
protocol must preserve backend-specific domains and error budgets.

The linear RK4 backend was then tested with five seeds and an OOD step-size range
`0.41--0.80`. OOD maximum absolute error averaged `1.67e-3`, while all values
and gradients remained finite. At `t=5`, repeated steps of `0.05` and `0.20`
gave mean errors `1.48e-9` and `4.23e-7`, respectively. This is a boundary
diagnostic: the protocol records the accuracy cost of larger steps instead of
turning successful execution into an unconditional reliability claim.

To test whether the same protocol survives a nonlinear ODE family, we also
validated a Logistic-equation RK4 step. Across five seeds and OOD step sizes
`0.26--0.50`, the maximum absolute error averaged `7.52e-6`, the directional
gradient relative error averaged `4.41e-11`, and long-horizon errors at `t=10`
were `9.82e-10` and `2.65e-7` for step sizes `0.05` and `0.20`. Module
gradients remained finite. Parameter calibration was less accurate
(`0.0260 +/- 0.0246` L1 error), which is retained as a positive difficulty
finding rather than hidden by the protocol.

## Public-data transfer evidence

The protocol was also applied to two provenance-aware public datasets. On the
GeoTES pilot-scale thermocouple histories (Zenodo DOI
`10.5281/zenodo.18979098`), the first measured cycle was used for
identification and the second cycle for transfer. Across three seeds, the
DFSC-plus-residual model reached a cycle-2 relative error of `0.3619 +/-
0.0085`, compared with `0.3632 +/- 0.0071` for a pure MLP, while using 750
parameters instead of 1251. On the heated-steam temperature profiles (Zenodo
DOI `10.5281/zenodo.15064388`), the held-out-condition RMSE was `43.33 +/-
0.56 K` for the direct DFSC model, `16.03 +/- 0.14 K` for the hybrid, and
`13.25 +/- 1.45 K` for the pure MLP. The latter result is intentionally
reported as a negative task-level result: a reliable primitive does not imply
that its hybrid composition is the best predictor for every dataset. These
experiments support real-data transfer and scope diagnosis, not universal
physical identification or unconditional predictive superiority.

The GeoTES experiment was also repeated with uniform subsampling over the
full first cycle, so that reducing the observation budget did not introduce a
different extrapolation horizon. With 40% of the observations, the
DFSC-plus-residual model reached `0.3607 +/- 0.0082` cycle-2 relative error,
versus `0.3657 +/- 0.0084` for the pure MLP. At 20%, 60%, and 100% coverage,
the hybrid did not improve the MLP. We therefore report this as a conditional
advantage interval rather than a monotone few-shot claim.

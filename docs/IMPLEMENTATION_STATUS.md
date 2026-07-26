# Implementation Status

## Implemented

- Differentiable one-parameter Mittag-Leffler evaluator.
- Explicit custom backward for the truncated series branch.
- History-free spectral layer with differentiable `alpha` and `beta`.
- 1D and 2D Dirichlet spectral constructors.
- Stable preset via `MLSLConfig.stable()` for broader negative-real spectral
  regimes.
- 1D Neumann spectral constructor in addition to 1D/2D Dirichlet constructors.
- 1D periodic and mixed-boundary spectral constructors.
- 2D Neumann, periodic, and mixed-boundary tensor-product constructors.
- Batched forward evaluation for dataset-level use.
- Baselines and experiment suite covering FNO, DeepONet, fPINN, OOD alpha,
  sparse inverse recovery, 2D inverse recovery, and runtime scaling.
- Unit tests for layer shapes, gradients, hybrid finite values, and 2D use.
- Examples for quickstart, batched evaluation, and inverse alpha/beta recovery.
- Primitive generality matrix across dimensions, fractional orders, initial
  conditions, batching, gradients, and inverse recovery.
- Paper-grade extension suite covering long horizons, repeated seeds, batch and
  device profiling, stronger neural baselines, forcing, fPINN repetition, and
  proposition-level timing evidence.
- Manufactured-solution validation for the forced MLSL component in a multi-mode
  spectral regime.
- Boundary-condition generality checks for 1D Dirichlet and Neumann settings.
- Extended boundary-condition generality checks for 1D Dirichlet, Neumann,
  periodic, and mixed settings.
- High-precision reference checks for the hybrid two-parameter Mittag-Leffler
  evaluator.
- Shifted-spectrum linear reaction-diffusion family experiment.
- Semilinear cubic-reaction backbone experiment using MLSL as the linear
  history-free primitive.
- Larger multi-seed neural baseline experiment.
- CUDA validation on RTX 5070, including CPU/GPU consistency, batch profiling,
  and half-precision capability probes.
- Diagnostic one-/two-parameter Mittag-Leffler evaluation with embedded
  truncation disagreement and branch counts.
- True direct/stable algorithm reconfiguration plus automatic regime selection.
- Generalized symmetric stiffness/mass spectral problems with mass projection.
- Differentiable constant-order linear Caputo L1 history fallback.
- Dense symmetric-PSD Lanczos Mittag-Leffler matrix-function action, including
  batched initial states, automatic size-based selection, and empirical
  subspace-disagreement diagnostics.
- Sparse tensor and matrix-free self-adjoint operator paths, including
  differentiable operator parameters, `N=4096` validation, and CUDA execution.
- FFT-accelerated Caputo-L1 full-trajectory history convolution, including
  direct-reference agreement, analytic power-law convergence, gradients,
  `N=65536` execution, and CPU/GPU consistency.
- Controlled complex-argument Mittag-Leffler evaluation and Arnoldi actions for
  general operators, validated against mpmath, high-precision matrix series,
  finite differences, matrix exponentials, and CUDA complex128.
- Domain-oriented application cases for regular-domain anomalous diffusion,
  assembled finite-element relaxation, graph memory diffusion, and controlled
  periodic fractional advection--diffusion. Each case reports its recommended
  algorithm, differentiable parameters, assumptions, and limitations.
- Experimental H-actin SPT evidence chain with 3,112 trajectories,
  trajectory-level leakage control, traditional and neural baselines, five
  splits, and an MLSL-residual regression component. Raw-data redistribution
  remains pending because the source page does not state a license.
- Semilinear mild-form Picard solver with convergence diagnostics, non-success
  retcodes, and autograd through fractional and nonlinear parameters.
- Single-package `dfsc` source layout; the legacy `mlsl` distribution has been removed.
- Machine-readable numerical reliability reports and strict validated-domain evaluation.
- A 20-check pre-release internal readiness gate covering numerical, API, test,
  application, documentation, and packaging evidence.

## Still Needed For Paper-Grade Primitive

- More accurate stable Mittag-Leffler evaluator with reference validation over
  wider long-time and stiff regimes.
- Contour or rational evaluation beyond the current conservative complex
  radius, including non-normal and pseudospectral error analysis.
- Wider reference validation for the hybrid two-parameter Mittag-Leffler
  evaluator in stiffer forced dynamics and larger negative arguments.
- Larger GPU profiling for full baseline training workloads.
- Broader nonlinear or variable-coefficient PDE families beyond linear spectral
  cases.
- Broader nonlinear solvers beyond the current fixed-point mild formulation.
- Block/preconditioned Krylov actions, contour methods, and online/adaptive
  fast-memory CQ/SOE time-stepping methods.
- Larger benchmark datasets with repeated seeds and confidence intervals.
- Cleaner narrative comparing MLSL as a primitive against black-box operator
  learners and residual-only fPINNs.

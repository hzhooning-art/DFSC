# dfsc

Research code for:

**A differentiable fractional scientific-computing environment for Mittag-Leffler spectral learning**

`dfsc` is the only public library name. The current research-grade beta provides a
Mittag-Leffler spectral primitive together with workflow helpers, component
discovery, diagnostics, tests, examples, and reproducible paper assets.

The core primitive implements a history-free spectral evolution operator:

```text
u(t) = Phi diag(E_alpha(-c^2 lambda_n^(beta/2) t^alpha)) Phi.T u0
```

Both `alpha` and `beta` can be optimized by backpropagation.

## Ecosystem Map

Implemented components:

- `dfsc.build_mlsl`: unified constructor for the MLSL primitive.
- `dfsc.build_operator_mlsl`: construct a spectral primitive from a symmetric PSD operator.
- `dfsc.build_generalized_operator_mlsl`: construct a mass-aware primitive from assembled stiffness/mass matrices.
- `dfsc.build_graph_mlsl`: construct a spectral primitive from a graph Laplacian.
- `dfsc.MLSLKrylov`: Lanczos matrix-function action without a full operator eigendecomposition.
- `dfsc.MLSLAdaptive`: tolerance-driven Lanczos propagation with an inspectable work history.
- `dfsc.MLSLArnoldi`: controlled action for non-self-adjoint and complex operators.
- `dfsc.SelfAdjointLinearOperator`: sparse and differentiable matrix-free `matvec` contract.
- `dfsc.evaluate_mittag_leffler`: autograd-preserving values with numerical-routing diagnostics.
- `dfsc.evaluate_mittag_leffler_adaptive`: adaptive truncation selected by successive detached disagreement.
- `dfsc.ReliabilityReport`: validated-domain, convergence, gradient, and error-indicator contract.
- `dfsc.ErrorBudget`: evaluator/Krylov/projection tolerance allocation that keeps missing estimates explicit.
- `dfsc.PreparedLanczosBasis`: reusable batched reduced spaces for repeated time/order queries.
- `dfsc.local_identifiability`: Hessian, covariance, standard-error, and correlation diagnostics.
- `dfsc.solve`: automatic problem--algorithm--solution interface for direct and history-aware paths.
- `dfsc.CaputoL1Problem`: first differentiable history-aware fallback for linear Caputo systems.
- `dfsc.caputo_l1_derivative_fft`: FFT-accelerated differentiable full-trajectory history operator.
- `dfsc.ForcedMittagLefflerSpectralLayer`: Duhamel-style forced wrapper.
- `dfsc.SemilinearSpectralProblem`: mild-form Picard iteration for state-dependent nonlinearities.
- `dfsc.make_trainable_orders`: bounded trainable `alpha/beta` variables.
- `dfsc.HybridResidualModel`: compose an MLSL backbone with a neural residual head.
- `dfsc.mlsl_applicability_report`: explicit scope check for retained spectral workloads.
- `dfsc.validate_dataset_manifest`: provenance-first contract for public benchmark data.
- `dfsc.application_catalog`: scoped application templates and their assumptions.
- `dfsc.MittagLefflerResidualRegressor`: measured relaxation/scattering backbone with a gated neural correction.
- `dfsc.load_anomdiffdb_mat`: provenance-aware experimental SPT benchmark loader.
- `dfsc.component_summary`: machine-readable component registry.

Experimental extension points include `dfsc.VariableOrderMLSL` and
`dfsc.DistributedOrderMLSL`. They compose the MLSL backbone over query-dependent
or quadrature-sampled orders and are labeled experimental because they are not
full variable-order or distributed-order fractional solvers.

## Application Strengths

The tested application layer focuses on model classes where a known
Mittag-Leffler propagator is a computational advantage:

- anomalous diffusion on regular 1D/2D domains;
- mass-aware linear relaxation from assembled stiffness/mass matrices;
- memory diffusion on undirected networks;
- controlled non-self-adjoint periodic advection--diffusion.

Use `dfsc.application_catalog()` to inspect assumptions and limitations, then
construct a configured `ApplicationCase` with `anomalous_diffusion_case`,
`assembled_relaxation_case`, `network_diffusion_case`, or
`advection_diffusion_case`. See [the application guide](docs/APPLICATION_DOMAINS.md).

The real-data suite now covers H-actin anomalous diffusion, two empirical
Brownian bead populations, and four CC BY 4.0 geomembrane stress-relaxation
tests. Results are deliberately mixed: the structured primitive helps on the
H-actin and fast-Brownian conditions, while the slow-Brownian condition favors
a stretched exponential; on geomembranes, residual composition improves the
mean long-time error while the bare fractional-Zener model does not. See [the
full protocol and interpretation boundary](docs/REAL_DATA_EVIDENCE.md).

A CC BY 4.0 pilot-scale thermal-storage benchmark additionally tests forced
multi-channel identification across two measured charging--discharging cycles.
The bare fractional model improves on its integer-order counterpart, while a
smaller DFSC-residual model reaches error parity with the larger pure MLP. This
result supports parameter-efficient composition rather than universal neural
superiority.

A fourth real-data domain uses 16 heated-steam injection experiments with ten
documented depth coordinates. Four extreme conditions are held out by design:
the largest flow, inlet temperature, column height, and water content. Bare
DFSC only slightly improves on the integer propagator. A pure MLP has the lowest
held-out RMSE, while the DFSC--residual model uses 1,360 rather than 4,673
parameters and reaches 15.98 K rather than 13.49 K RMSE. This is a
parameter-efficiency tradeoff and a recorded applicability boundary, not an
accuracy win.

## Install

The release candidate can be installed from a built wheel:

```bash
python -m pip install dist/dfsc-0.1.0rc1-py3-none-any.whl
```

After the public release, the standard installation command will be:

```bash
python -m pip install dfsc
```

For development from this folder:

```bash
python -m pip install -e ".[test,docs]"
```

Run the complete unit-test suite with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Minimal validation gate:

```bash
python validate.py
```

Runtime capability report:

```bash
python tools/mlsl_doctor.py
```

Core reproducibility gate:

```bash
python tools/reproduce_core.py
```

Current maturity report:

```bash
python tools/maturity_report.py
```

Pre-release internal readiness gate:

```bash
python -B tools/pre_release_audit.py
```

## Quick Use

```python
import torch
from dfsc import MLSLConfig, build_dirichlet_mlsl_1d

x, layer = build_dirichlet_mlsl_1d(
    num_points=128,
    num_modes=24,
    config=MLSLConfig.stable(terms=120),
)

u0 = torch.sin(torch.pi * x)
times = torch.linspace(0.0, 0.05, 10)
alpha = torch.tensor(1.35, requires_grad=True)
u = layer(u0, times, alpha, beta=torch.tensor(1.5))
loss = u[-1].square().mean()
loss.backward()
```

Unified constructor:

```python
from dfsc import MLSLConfig, build_mlsl

x, layer = build_mlsl(
    dimension=1,
    boundary="periodic",
    num_points=128,
    num_modes=31,
    config=MLSLConfig.stable(),
)
```

Component registry:

```python
import dfsc

print(dfsc.component_summary())
print(dfsc.algorithm_registry())
print(dfsc.ecosystem_gap_report())
```

Applicability check:

```python
report = dfsc.mlsl_applicability_report(layer.eigenvalues, layer.eigenvectors)
assert report.supported, report.unsupported_reasons
```

Problem--algorithm--solve interface:

```python
problem = dfsc.FractionalSpectralProblem(
    layer=layer,
    u0=u0,
    times=times,
    alpha=alpha,
    beta=beta,
)
solution = dfsc.solve(problem)  # AutoDFSC selects direct or stable evaluation.
print(solution.algorithm, solution.diagnostics)
loss = solution.final.square().mean()
loss.backward()
```

History-aware fallback when a retained propagator is not supplied:

```python
problem = dfsc.CaputoL1Problem(
    operator=A,
    u0=u0,
    alpha=alpha,
    final_time=1.0,
    num_steps=200,
)
solution = dfsc.solve(problem)
```

Larger dense symmetric operators can avoid a full eigendecomposition:

```python
problem = dfsc.OperatorSpectralProblem(A, u0, times, alpha, beta=beta)
solution = dfsc.solve(problem, dfsc.MLSLKrylov(krylov_dimension=48))
print(solution.diagnostics["embedded_relative_disagreement"])
```

Tolerance-driven operator action:

```python
solution = dfsc.solve(
    problem,
    dfsc.MLSLAdaptive(
        dimension_schedule=(8, 16, 24, 32, 48, 64),
        rtol=1e-6,
        atol=1e-9,
    ),
)
print(solution.diagnostics["selected_krylov_dimension"])
print(solution.diagnostics["adaptive_converged"])
```

The stopping indicator is a successive-action disagreement, not a rigorous
global error bound. Use `strict=True` to reject an exhausted schedule.

Semilinear mild-form iteration uses the same spectral backbone:

```python
problem = dfsc.SemilinearSpectralProblem(
    layer, u0, times, alpha, nonlinearity=lambda u: -0.02 * u**3
)
solution = dfsc.solve(problem, dfsc.MLSLPicard(tolerance=1e-7))
```

Matrix-free propagation requires only an operator action:

```python
operator = dfsc.SelfAdjointLinearOperator(
    size=n,
    matvec=lambda vector: apply_discrete_operator(vector),
    dtype=u0.dtype,
    device=u0.device,
)
problem = dfsc.LinearOperatorSpectralProblem(operator, u0, times, alpha)
solution = dfsc.solve(problem)  # AutoDFSC selects MLSLKrylov.
```

Fast full-trajectory Caputo-L1 residual evaluation:

```python
problem = dfsc.CaputoHistoryProblem(
    values=predicted_trajectory,
    alpha=alpha,
    final_time=1.0,
)
derivative = dfsc.solve(problem)  # Direct for short histories, FFT for long histories.
loss = derivative.values.square().mean()
loss.backward()
```

Controlled non-self-adjoint or complex propagation:

```python
problem = dfsc.GeneralOperatorProblem(A, u0, times, alpha)
solution = dfsc.solve(problem, dfsc.MLSLArnoldi())
print(solution.diagnostics["observed_reduced_radius"])
```

The first complex path is intentionally limited to scalar `|z| <= 4` and
Arnoldi reduced radius `<= 4`. Larger complex regimes require a future contour
or rational evaluator.

## Run

From this folder:

```bash
python experiments/exp01_forward_demo.py
python experiments/exp02_gradient_check.py
python experiments/exp03_alpha_recovery.py
python experiments/exp04_beta_gradient_check.py
python experiments/exp05_alpha_beta_recovery.py
python experiments/exp06_runtime_scaling.py
python experiments/exp07_custom_backward_check.py
python experiments/exp08_stable_evaluator_scan.py
python experiments/exp09_reference_accuracy.py
python experiments/exp10_long_time_prediction.py
python experiments/exp11_noise_robustness.py
python experiments/exp12_mode_sensitivity.py
python experiments/exp13_hybrid_threshold_scan.py
python experiments/exp14_fno_dataset_long_time.py
python experiments/exp15_sparse_observation_inverse.py
python experiments/exp16_deeponet_dataset_long_time.py
python experiments/exp17_ood_alpha_generalization.py
python experiments/exp18_2d_forward_inverse.py
python experiments/exp19_fpinn_scalar_inverse.py
python experiments/exp20_batch_scaling.py
python experiments/exp21_primitive_generality.py
python experiments/exp22_paper_grade_extensions.py
python experiments/exp23_manufactured_forcing.py
python experiments/exp24_boundary_generality.py
python experiments/exp25_gap_closure.py
python experiments/exp26_2d_nonlinear_extensions.py
python experiments/exp27_gpu_validation.py
python experiments/exp36_krylov_semilinear_validation.py
python experiments/exp37_sparse_matrix_free_validation.py
python experiments/exp38_fast_history_validation.py
python experiments/exp39_complex_arnoldi_validation.py
python experiments/exp40_application_domain_validation.py
python tools/fetch_anomdiffdb.py
python experiments/exp41_real_spt_evidence_chain.py
```

For a reproducible machine-readable suite:

```bash
python experiment_suite.py
```

This writes CSV/JSON outputs to:

```text
results/
  summary.json
  tables/
```

## Current Scope

Current primitive scope:

- 1D Dirichlet Laplacian basis on `(0, 1)`.
- 1D Neumann Laplacian basis.
- 1D periodic and mixed-boundary Laplacian bases.
- 2D tensor-product Dirichlet Laplacian basis.
- 2D tensor-product Neumann, periodic, and mixed-boundary Laplacian bases.
- Homogeneous fractional dynamics.
- One-parameter Mittag-Leffler function `E_alpha(z)`.
- Power-series evaluation with explicit custom backward.
- First hybrid evaluator for non-positive real large-argument regimes.
- Stable configuration preset for broader real-negative spectral regimes.
- Batched calls for dataset-level SciML workflows.
- First forced-dynamics extension using a two-parameter Mittag-Leffler kernel.
- Automatic direct/stable selection with inspectable detached diagnostics.
- Symmetric stiffness/mass generalized eigenproblems with correct mass projection.
- Constant-order linear Caputo L1 fallback for `0 < alpha < 1`.
- Direct and FFT-accelerated Caputo-L1 full-trajectory derivative evaluation.
- Fully reorthogonalized Lanczos propagation for dense symmetric PSD operators.
- Sparse COO/CSR-compatible and matrix-free self-adjoint operator actions.
- Moderate-radius complex Mittag-Leffler evaluation and general-operator Arnoldi actions.
- Mild-form Picard iteration for semilinear problems on a supplied MLSL basis.

The embedded disagreement reported by `evaluate_mittag_leffler` compares two
truncation levels. It is generally a diagnostic, not a rigorous global error
bound. For real `z <= 0`, `0 < alpha <= 1`, and a truncation already in the
decreasing alternating-term regime, dfsc instead reports the first omitted term
as a certified local series remainder bound. This certificate does not cover
hybrid-branch, Krylov, or spatial-projection error.
The analogous Krylov diagnostic compares two subspace dimensions and has the
same non-rigorous status.

Repeated queries for a fixed operator and initial-state batch can reuse a
prepared Lanczos basis:

```python
prepared = dfsc.prepare_lanczos_basis(A, u0_batch, krylov_dimension=32)
values = dfsc.apply_prepared_lanczos_basis(prepared, times, alpha, beta=beta)
```

The prepared basis remains differentiable with respect to query-time `alpha`
and `beta`; rebuild it after changing `A` or `u0_batch`.

See `docs/API.md` and `docs/IMPLEMENTATION_STATUS.md` for details.

The current repository also contains a release checklist, CI workflow, and
MkDocs skeleton. These files prepare dfsc for a versioned public package, but a
PyPI release and hosted API reference are still future release steps.

Public physical evidence is tracked through manifests, checksums, preprocessing
rules, and per-run outputs. The current suite covers three AnomDiffDB conditions
and four CC BY 4.0 geomembrane stress-relaxation tests. The AnomDiffDB source
does not state a redistribution license, so those raw files remain excluded
from release artifacts.

## Experiments

- `exp01_forward_demo.py`: checks direct spectral forward evaluation.
- `exp02_gradient_check.py`: finite-difference check for `alpha`.
- `exp03_alpha_recovery.py`: recovers hidden `alpha` from synthetic observations.
- `exp04_beta_gradient_check.py`: finite-difference check for `beta`.
- `exp05_alpha_beta_recovery.py`: jointly recovers hidden `alpha` and `beta`.
- `exp06_runtime_scaling.py`: compares direct MLSL evaluation with a differentiable L1 time-marching baseline for scalar Caputo relaxation.
- `exp07_custom_backward_check.py`: compares explicit custom backward with PyTorch autograd and finite differences.
- `exp08_stable_evaluator_scan.py`: scans the series and first hybrid evaluator on large negative inputs.
- `exp09_reference_accuracy.py`: checks series/hybrid values against a high-precision mpmath series on a controlled interval.
- `exp10_long_time_prediction.py`: compares MLSL long-time extrapolation with a black-box MLP neural field.
- `exp11_noise_robustness.py`: sweeps observation noise for hidden alpha/beta recovery.
- `exp12_mode_sensitivity.py`: tests spectral truncation sensitivity.
- `exp13_hybrid_threshold_scan.py`: checks hybrid switching threshold sensitivity against controlled reference points.
- `exp14_fno_dataset_long_time.py`: compares MLSL with a minimal FNO baseline on dataset-level long-time prediction.
- `exp15_sparse_observation_inverse.py`: recovers alpha/beta from sparse sensors and sparse times.
- `exp16_deeponet_dataset_long_time.py`: compares MLSL with a minimal DeepONet baseline.
- `exp17_ood_alpha_generalization.py`: tests alpha OOD generalization.
- `exp18_2d_forward_inverse.py`: verifies the primitive on a small 2D tensor-product spectral basis.
- `exp19_fpinn_scalar_inverse.py`: adds a residual-based fPINN-style scalar inverse baseline for ``0 < alpha < 1``.
- `exp20_batch_scaling.py`: tests batched MLSL calls as a reusable primitive.
- `exp21_primitive_generality.py`: checks primitive generality across dimensions, fractional orders, inputs, batching, gradients, and inverse recovery.
- `exp22_paper_grade_extensions.py`: extends the evidence with long-horizon validation, repeated seeds, profiling, stronger neural baselines, forcing, and proposition-level checks.
- `exp23_manufactured_forcing.py`: validates the forced MLSL component against manufactured analytic modal solutions.
- `exp24_boundary_generality.py`: validates MLSL across Dirichlet and Neumann 1D boundary conditions.
- `exp25_gap_closure.py`: closes pre-writing gaps with periodic/mixed boundaries, E_ab reference checks, profiling, reaction-diffusion, and larger neural baseline seeds.
- `exp26_2d_nonlinear_extensions.py`: validates 2D Neumann/periodic/mixed boundaries and a semilinear reaction backbone workflow.
- `exp27_gpu_validation.py`: validates CUDA execution, CPU/GPU consistency, batch profiling, and half-precision support.
- `exp36_krylov_semilinear_validation.py`: validates Lanczos convergence and gradients against full eigendecomposition, semilinear Picard convergence, and CUDA execution.
- `exp37_sparse_matrix_free_validation.py`: validates dense/sparse/matrix-free agreement, trainable operator parameters, large matrix-free execution, and CUDA gradients.
- `exp38_fast_history_validation.py`: validates FFT history values and gradients, analytic L1 convergence, measured scaling, large trajectories, and CUDA consistency.
- `exp39_complex_arnoldi_validation.py`: validates complex values against mpmath, Arnoldi against high-precision matrix series, gradients, non-normal diagnostics, and CUDA complex execution.
- `exp40_application_domain_validation.py`: validates four scoped application templates through order/physical-parameter gradients, mass projection, graph invariance, and the alpha=1 matrix-exponential limit.
- `exp41_real_spt_evidence_chain.py`: compares traditional, direct MLSL, pure-neural, and MLSL-residual models on held-out experimental trajectories over five trajectory-level splits.
- `exp42_external_fractional_solver_benchmark.py`: compares adaptive direct queries with FDEint and pycaputo on scalar and coupled linear Caputo systems.
- `exp43_real_brownian_two_population.py`: evaluates two empirical Brownian-bead populations over five trajectory splits.
- `exp44_real_geomembrane_relaxation.py`: evaluates four experimental stress-relaxation conditions over three seeds.
- `exp45_adaptive_krylov_calibration.py`: calibrates tolerance-driven Krylov work and dense-reference action error on GPU.
- `exp49_error_budget_validation.py`: validates the restricted alternating-series certificate.
- `exp50_sample_ood_matrix.py`: measures sample efficiency and structured OOD behavior.
- `exp51_real_heated_steam.py`: defines a spatial condition-OOD protocol for public heated-steam data.
- `exp52_inverse_identifiability.py`: reports multi-start recovery and local alpha/beta uncertainty.
- `exp53_krylov_reuse_benchmark.py`: benchmarks repeated-query basis reuse on CPU and GPU.

The series implementation is suitable for small/moderate `|z|` experiments. Large-time or stiff regimes should use `MLSLConfig.stable()`.

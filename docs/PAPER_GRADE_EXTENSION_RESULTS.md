# Paper-Grade Extension Results

This document summarizes the current extension experiments beyond the initial
primitive validation.

Run:

```bash
python experiments/exp22_paper_grade_extensions.py
```

Outputs:

```text
results/paper_grade_extension_summary.json
results/tables/long_horizon_stability_accuracy.csv
results/tables/multi_seed_inverse.csv
results/tables/batch_and_device_profile.csv
results/tables/stronger_operator_baselines.csv
results/tables/fpinn_repeated_baseline.csv
results/tables/forcing_quadrature_validation.csv
results/tables/manufactured_forcing_validation.csv
results/tables/proposition_evidence.csv
results/tables/eab_reference_accuracy.csv
results/tables/mixed_precision_gpu_profile.csv
results/tables/reaction_diffusion_family.csv
results/tables/boundary_extended_matrix.csv
results/tables/larger_neural_baseline_multiseed.csv
```

## Current Summary

- Restricted negative-real series certificate:
  - certified cases: `59/60` (one case had not entered the decreasing-term regime);
  - certified-case reference coverage: `100%`;
  - median bound/error effectivity for resolved errors: `1.246`.
- Local alpha/beta identifiability matrix:
  - cases locally full-rank: `27/27`;
  - mean absolute alpha error: `6.89e-03`;
  - mean absolute beta error: `3.64e-03`;
  - maximum Hessian condition number: `8.67`.
- Prepared Lanczos repeated-query benchmark:
  - exact agreement with the original fixed-path implementation in `18/18` cases;
  - query-only CPU speedup: `4.61x--7.11x`;
  - query-only RTX 5070 speedup: `13.07x--16.22x`;
  - one-time basis preparation is excluded from these speedups.
- Controlled sample/OOD matrix:
  - nested training sets: `16, 32, 64, 128`;
  - three seeds and five test regimes;
  - at 16 samples, joint-OOD error falls from `0.545` for the pure MLP to
    `0.0480` for DFSC plus residual learning;
  - at 128 samples, the corresponding errors are `0.204` and `0.0346`.
- Heated-steam spatial condition OOD:
  - source rows: `32,370`; retained train/test rows: `5,090/1,470`;
  - pure MLP: `13.49 +/- 1.56 K` RMSE with `4,673` parameters;
  - DFSC plus residual: `15.98 +/- 0.63 K` with `1,360` parameters;
  - bare DFSC is only marginally better than the matched integer model.

- Long-horizon stability pass rate: `1.0`.
- Multi-seed inverse recovery:
  - alpha relative error mean: `1.28e-05`.
  - alpha relative error std: `1.27e-05`.
  - beta relative error mean: `2.42e-05`.
  - beta relative error std: `2.53e-05`.
- GPU available in the current run: `false`.
- CPU batch profile at batch size 512: `1.02e-03` seconds.
- Stronger operator baselines on long-time extrapolation:
  - MLSL oracle mean relative error: `0.0`.
  - FNO mean relative error: `3.74`.
  - DeepONet mean relative error: `2.05`.
- Repeated fPINN scalar inverse baseline:
  - alpha relative error mean: `5.47e-02`.
  - solution relative error mean: `2.94e-02`.
- Nonhomogeneous forcing quadrature:
  - q=12 relative error: `2.09e-04`.
  - q=96 relative error: `9.21e-06`.
  - convergence ratio q12/q96: `22.73`.
- Manufactured forcing validation:
  - q=32 mean relative error: `9.21e-03`.
  - q=64 mean relative error: `5.28e-03`.
  - q=128 mean relative error: `3.05e-03`.
  - q=128 max relative error: `6.88e-03`.
  - all tested outputs and gradients finite: `true`.
- Boundary-condition generality:
  - Dirichlet/Neumann/Periodic/Mixed total checks: `45`.
  - passed checks: `45`.
  - pass rate: `1.0`.
- Two-parameter Mittag-Leffler reference accuracy:
  - max relative error: `1.30e-06`.
  - max absolute error: `1.07e-08`.
- Reaction-diffusion family:
  - pass rate: `1.0`.
- 2D boundary extension:
  - Neumann/periodic/mixed pass rate: `1.0`.
- Semilinear cubic-reaction backbone:
  - pass rate: `1.0`.
- Larger neural baseline multi-seed long-time extrapolation:
  - DeepONet mean relative error: `1.95`, std: `0.29`.
  - FNO mean relative error: `3.31`, std: `1.24`.
- Mixed precision / GPU profiling:
  - CUDA available in current environment: `true`.
  - GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`.
  - PyTorch: `2.11.0+cu128`.
  - CUDA: `12.8`.
  - CPU/GPU max output relative error: `3.36e-07`.
  - CPU/GPU max alpha-gradient relative error: `5.26e-07`.
  - CPU/GPU max beta-gradient relative error: `5.43e-07`.
  - GPU batch 1024 float64 time: `3.22e-03` seconds.
  - GPU batch 1024 float32 time: `2.32e-03` seconds.
  - all GPU finite checks: `true`.
- Proposition-level history-free evidence:
  - L1 over MLSL speedup at 400 time steps: `621.66x`.

## Interpretation

The extension experiments strengthen the primitive-level claim in six ways:

1. Longer horizons remain numerically stable in the tested regimes.
2. Multi-seed inverse recovery has very small mean and variance.
3. The current machine has no CUDA device, but CPU batch scaling is already fast.
4. Stronger FNO and DeepONet baselines still struggle with long-time
   extrapolation, supporting the structured-kernel narrative.
5. A first forced MLSL extension using the two-parameter Mittag-Leffler kernel
   shows quadrature convergence and now has a multi-mode manufactured-solution
   check.
7. The primitive now covers Dirichlet, Neumann, periodic, and mixed 1D boundary
   conditions in the validation matrix.
8. The two-parameter hybrid evaluator has direct high-precision reference
   checks in controlled regimes.
9. A shifted-spectrum reaction-diffusion family verifies that MLSL can represent
   linear PDE families beyond pure fractional diffusion.
10. 2D tensor-product boundary coverage now extends beyond Dirichlet to
    Neumann, periodic, and mixed settings.
11. A semilinear reaction experiment verifies that MLSL can serve as a
    history-free linear backbone inside nonlinear workflows.
6. The history-free complexity claim has direct timing evidence against an L1
   history-marching baseline.

## Remaining Gaps


- Repeat the extension suite with CUDA hardware.
- Add confidence intervals over more seeds for the neural baselines.
- Add stronger tuned baselines if reviewer-facing comparisons require them.
- Extend GPU profiling to larger grids and full neural baselines.
- Extend boundary-condition coverage to 2D Neumann/periodic/mixed cases.
- Replace the semilinear backbone smoke test with full nonlinear solver
  benchmarks and manufactured nonlinear references.
- Extend from Dirichlet spectral domains to additional boundary conditions and
  PDE families.

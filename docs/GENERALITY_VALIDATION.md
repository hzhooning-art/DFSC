# Generality Validation

The current MLSL component can be tested for primitive-level generality, but the
claim should be stated carefully.

## What We Validate

The validation matrix checks whether the same component works across:

- 1D and 2D Dirichlet spectral domains.
- Subdiffusive and superdiffusive temporal orders.
- Multiple spatial fractional orders.
- Low-mode, multi-mode, localized, and batched initial conditions.
- Forward evaluation, differentiability with respect to `alpha` and `beta`,
  linearity, batch consistency, and inverse recovery.

Run:

```bash
python experiments/exp21_primitive_generality.py
```

Outputs:

```text
results/tables/primitive_generality_matrix.csv
results/primitive_generality_summary.json
```

## Current Result

The current stable-configuration run passes 34 of 34 checks.

This supports the claim that MLSL is already a reusable SciML primitive in
moderate regimes: it is not limited to one initial condition, one parameter
choice, one dimension, or one task type.

## Observed Boundary

Earlier default-series runs exposed an important limitation rather than
invalidating the direction:

- For `alpha=0.65, beta=2.0`, the default power-series branch gives finite
  forward outputs but non-finite gradients in some stiff settings.
- Joint recovery of `alpha` and `beta` from a single low-frequency mode is weakly
  identifiable, so the validation now tests alpha-only recovery for single-mode
  data and joint recovery for multi-mode data.

## Interpretation

The stable configuration has demonstrated broad empirical portability across
the current validation matrix. The next primitive-level milestone is to expand
this matrix to harder PDE families, larger batches, longer times, GPU execution,
and repeated random seeds.

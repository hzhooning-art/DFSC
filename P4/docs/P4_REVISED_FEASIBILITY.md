# P4 Revised Route: Basic Feasibility Result

## Scope

This experiment tests the revised P4 hypothesis:

> A known Mittag-Leffler spectral propagator can serve as a differentiable,
> parameter-conditioned backbone for learning a family of fractional dynamics,
> while a small neural residual accounts for a controlled model discrepancy.

It is a feasibility gate, not a final claim of broad operator-learning
generality. The experiment reuses the P1 `mittag_leffler_e` evaluator and its
Dirichlet spectral basis. The parameter family contains `alpha`, `beta`, and a
differentiable modal-rate scale `lambda`.

## Protocol

- Grid: 32 spatial points and 12 retained modes.
- Device: NVIDIA GeForce RTX 5070 Laptop GPU.
- Training region: `alpha in [0.76, 0.84]`, `beta in [1.85, 2.05]`,
  `lambda in [0.85, 1.15]`.
- OOD region: paired parameter corners with `alpha in {0.70, 0.90}`,
  `beta in {1.70, 2.30}`, and `lambda in {0.65, 1.35}`.
- Target: MLSL propagation plus a bounded correction
  `0.06 exp(-0.5 t) tanh(u0)`.
- Models: MLSL-only, pure conditional MLP, and hybrid residual model.
- Training: 24 samples, 120 Adam steps, float64 CUDA execution.

## Results

| Model | ID relative L2 | OOD relative L2 |
|---|---:|---:|
| MLSL-only | 0.3630 | 0.2313 |
| Pure conditional MLP | 0.2086 | 0.5464 |
| MLSL + residual | **0.0166** | **0.0274** |

The hybrid model reduces the controlled model-discrepancy error by roughly an
order of magnitude relative to the pure conditional MLP in both regions. The
OOD result is particularly useful as a feasibility signal: the fixed spectral
structure remains available outside the narrow training parameter band, while
the unconstrained MLP extrapolates poorly in this small-data setting.

## Differentiability and evaluator checks

At a held-out parameter point, the absolute output gradients were:

| Parameter | Absolute gradient |
|---|---:|
| `alpha` | 1.4967 |
| `beta` | 3.2522 |
| `lambda` | 2.7615 |

All three gradients were finite. The relative difference between the 40-term
and 80-term evaluator runs was 0.0 for this test regime.

## Decision

**P4 remains feasible as a research direction.** The evidence supports a
focused next stage: multi-seed parameter-family experiments, stronger pure
operator baselines, nontrivial forcing, and explicit P2-style error/OOD
audits. It does **not** yet justify claims that P4 is a general fractional
solver, a universal neural operator, or a demonstrated improvement over all
existing parameter-conditioned operator-learning methods.

The machine-readable result is stored in
`P4/results/p4_parameter_family_feasibility.json`, and the executable is
`P4/experiments/parameter_family_feasibility.py`.

# P4 Integrated Workflow: Calibration Gate

## Protocol

Thirty-two independent fractional relaxation tasks were generated with
off-grid `alpha` and `lambda` values, 32 observations, and Gaussian noise
`sigma=0.01`. Batched MLSL calibration optimized alpha and lambda jointly for
120 Adam steps using 16 series terms. A matched 41x41 high-accuracy grid search
was used as a baseline.

## Results

| method | alpha error | lambda error | joint error | elapsed time |
|---|---:|---:|---:|---:|
| batched differentiable MLSL | 0.00767 | **0.00371** | **0.00919** | 0.615 s |
| grid search | **0.00716** | 0.00639 | 0.01056 | **0.244 s** |

All tested gradients were finite. The raw result is stored in
`P4/results/p4_integrated_calibration.json`.

## Interpretation

The first calibration gate supports practical differentiability: MLSL can
jointly calibrate many tasks and gives a slightly lower joint error than the
grid baseline. It does **not** show a wall-clock advantage at this small task
size, because the grid baseline is highly vectorized on the GPU.

The defensible claim is therefore not “MLSL is always faster.” It is that MLSL
provides a differentiable calibration path that can be inserted into a larger
autograd workflow, while retaining competitive parameter recovery. Larger
batches, repeated optimization, and joint neural training are needed before
any throughput claim is made.

## Gate status

**Partial pass.** Continue to cross-module reuse experiments, but retain the
grid-search result as a mandatory baseline and report speed only for matched
workloads. The next experiment must test whether the same primitive remains
stable when used inside more than one SciML training objective.

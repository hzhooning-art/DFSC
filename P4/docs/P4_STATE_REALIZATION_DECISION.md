# P4 State-Realization Feasibility Decision

## Protocol

The probe fitted differentiable exponential-sum state coefficients over an
alpha/rate envelope and served 1000 irregular time queries for a batch of 256
trajectories. It compared 4, 8, 16, and 32 state components against direct
100-term MLSL on the RTX 5070 GPU. Forward values and final-query parameter
gradients were both measured.

## Results

| model | all-query value RMSE | final value RMSE | final gradient RMSE | elapsed time |
|---|---:|---:|---:|---:|
| direct MLSL reference | 0 | 0 | 0 | 0.062 s |
| 4-state realization | 2.58e-2 | 4.33e-2 | 0.406 | 0.199 s |
| 8-state realization | 1.62e-2 | 3.39e-2 | 0.225 | 0.155 s |
| 16-state realization | 3.88e3 | 1.18e4 | 8.15e4 | 0.157 s |
| 32-state realization | 1.74e4 | 5.12e4 | 3.65e5 | 0.173 s |

The raw result is stored in
`P4/results/p4_state_realization_feasibility.json`.

## Decision

The first state-realization construction fails the P4 hard stop:

- low state counts have unacceptable value and gradient error;
- increasing the state count causes an ill-conditioned exponential fit;
- all state variants are slower than direct batched MLSL in this workload;
- no long-horizon amortization advantage is visible at 1000 irregular queries.

This is not a reason to hide the route behind more degree or state-count
tuning. A completely different stable rational/state-space construction might
still be studied in a future project, but the present P4 route is **closed as
an independent paper direction**.

## Overall P4 conclusion

P4 has now failed four independent route gates: inverse mismatch detection,
active observation design, spectral compression, and state realization. The
remaining P1--P3 series should not be forced into a five-paper sequence. P4
artifacts remain useful as negative benchmarks and possible DFSC utilities, but
there is currently no evidence for a defensible independent P4 contribution.

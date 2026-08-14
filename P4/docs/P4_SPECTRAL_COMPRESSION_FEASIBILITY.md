# P4 Spectral Compression Feasibility Gate

## Controlled test

The first compression test approximated the scalar family
`E_alpha(-lambda t^alpha)` on a bounded alpha/rate/time domain with a
differentiable Chebyshev tensor representation. The test used 4096 random
off-grid points on an RTX 5070 and compared against a 100-term reference.

## Results

| representation | value RMSE | gradient RMSE | mean batch time |
|---|---:|---:|---:|
| MLSL, 8 terms | 1.77e-3 | 2.27e-2 | 0.304 ms |
| MLSL, 16 terms | **1.87e-7** | **6.19e-6** | **0.301 ms** |
| Chebyshev 4x4x4 | 3.12e-3 | 1.36e-2 | 1.337 ms |
| Chebyshev 6x6x6 | 1.38e-4 | 1.68e-3 | 1.805 ms |
| Chebyshev 8x8x6 | 1.38e-4 | 1.68e-3 | 2.050 ms |

The raw result is stored in
`P4/results/p4_spectral_compression_feasibility.json`.

## Decision

The scalar Chebyshev compression candidate fails the first practical Pareto
test. It is slower than the direct MLSL series and does not simultaneously
improve value and gradient accuracy. Its coefficient storage is small, but
that alone is not a sufficient advantage when the uncompressed evaluator has
lower latency and higher accuracy.

The result also weakens the innovation claim: a polynomial surrogate over a
bounded scalar family is not by itself a distinctive fractional-scientific
computing contribution.

## Scope after the gate

The current scalar compression implementation is frozen and should not be
advanced by tuning polynomial degrees. A final, separately scoped probe may
test whether compression becomes useful for large matrix/operator batches,
where direct matrix-function actions have a different cost profile. That probe
must compare wall time, memory, value error, and gradient error at matched
accuracy. If no Pareto improvement appears at matrix scale, the entire P4
compression route should be stopped as an independent paper.

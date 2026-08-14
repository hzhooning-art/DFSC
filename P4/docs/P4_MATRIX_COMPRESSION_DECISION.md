# P4 Matrix-Scale Compression Decision

## Probe

The final matrix-scale probe used a 256-dimensional symmetric spectral
operator, a batch of 512 vectors, 100-term Mittag-Leffler coefficients, and
irregular times on `[0.2, 2.0]`. The reference retained all 256 spectral modes.
Truncated representations retained 32, 64, 128, or 256 modes.

## Results

| retained rank | basis memory | value RMSE | gradient error | batch time |
|---:|---:|---:|---:|---:|
| 32 | 64 KiB | 0.4383 | 64.86 | 2.00 ms |
| 64 | 128 KiB | 0.3656 | 88.37 | 5.35 ms |
| 128 | 256 KiB | 0.2520 | 103.27 | 8.99 ms |
| 256 | 512 KiB | 0 | 0 | 16.00 ms |
| full reference | 512 KiB | 0 | 0 | 15.50 ms |

The raw result is stored in
`P4/results/p4_matrix_compression_probe.json`.

## Final route decision

The predefined retention rule required a lower-cost representation with value
RMSE at most `1e-4` and gradient error at most `1e-3`. No reduced rank met this
condition. The only accurate representation retained the full basis and did
not improve latency.

Therefore the P4 **spectral compression route is closed as an independent
paper direction**. The negative result is consistent across scalar and matrix
probes: the proposed compression has no demonstrated Pareto advantage over
the existing MLSL evaluation in the tested regimes.

The implementation and results remain useful as a DFSC benchmark, but further
rank tuning, polynomial-degree tuning, or larger synthetic examples should not
be pursued as P4 research unless a new approximation mechanism with a distinct
theoretical basis is introduced.

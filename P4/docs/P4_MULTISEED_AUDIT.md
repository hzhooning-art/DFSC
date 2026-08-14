# P4 Multi-seed and Audit Evidence

## Result

Five random seeds were evaluated on the same revised P4 parameter family.
The experiment used the P1 MLSL evaluator and a controlled target consisting
of MLSL propagation plus a bounded residual. It is intended to test statistical
stability, not to serve as a final external benchmark.

| Model | ID relative L2 (mean +/- std) | OOD relative L2 (mean +/- std) |
|---|---:|---:|
| Pure conditional MLP | 0.2232 +/- 0.0087 | 0.5023 +/- 0.0423 |
| MLSL + residual | **0.0169 +/- 0.0011** | **0.0243 +/- 0.0032** |

The hybrid model's mean OOD error is approximately 20.7 times lower than the
pure conditional MLP in this controlled small-data regime. The variance is
also lower for the hybrid model. This supports the feasibility of the revised
P4 composition mechanism, but does not establish universal OOD behavior.

## P2-style selection audit

Using OOD validation loss as the score, with the hybrid model required to win
both an early and a late diagnostic window:

- selected candidate: `hybrid_residual`;
- confidence: `0.9516`;
- temporal agreement: passed;
- accepted: `true`.

The confidence is an empirical validation margin, not a probability and not a
rigorous error bound.

## Next gate

The next experiment must leave the controlled residual setting: add a
nontrivial time-dependent forcing or coefficient perturbation, compare against
a stronger parameter-conditioned operator baseline, and report whether the
hybrid advantage survives across multiple seeds and parameter regions.

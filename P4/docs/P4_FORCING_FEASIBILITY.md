# P4 Forcing Feasibility Gate

The next gate adds a nontrivial time- and space-dependent forcing term to the
parameter-conditioned family. The target is

`u_MLSL + 0.045 sin(2.5 t) cos(pi x) + 0.025 exp(-0.7 t) tanh(u0^2)`.

This remains a controlled synthetic test. It is deliberately not presented as
an external physical validation.

## Five-seed result

| Model | ID relative L2 (mean +/- std) | OOD relative L2 (mean +/- std) |
|---|---:|---:|
| Pure conditional MLP | 0.2745 +/- 0.0082 | 0.5197 +/- 0.0498 |
| MLSL + residual | **0.0585 +/- 0.0015** | **0.1537 +/- 0.0048** |

The hybrid model retains a mean OOD improvement factor of 3.38 in this harder
setting. Its variance remains smaller than the pure neural baseline. The drop
from the approximately 20x improvement seen for the simpler residual is
important: it shows that the hybrid advantage depends on the forcing regime
and should not be reported as universal.

## Decision

The revised P4 route remains feasible, but the next required test is a genuine
operator-learning comparison with forcing supplied as an input function, not
only a coordinate-level synthetic correction. The final claim should be
restricted to parameter-conditioned fractional families for which the
Mittag-Leffler structure is informative.

Result file: `P4/results/p4_parameter_family_forcing.json`.
Executable: `P4/experiments/parameter_family_forcing.py`.

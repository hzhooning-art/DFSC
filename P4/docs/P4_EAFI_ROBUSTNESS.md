# P4 EAFI Robustness Stress Test

## Setup

This stress test uses an off-grid truth (`alpha=0.773`, `rate=0.637`) and a
noise scale that is not supplied to the inference rule. The evaluator uses
eight terms, with the multiplier `c=0.25` carried over from the hold-out
experiment. Noise is estimated from a robust scale of first differences. A
second test contaminates 10% of the forward signal with a stretched
exponential, creating a near-miss rather than an obvious structural failure.

## Results

| test | replicates | coverage / acceptance | mean normalized area or loss |
|---|---:|---:|---:|
| off-grid, ordinary profile | 40 | 1.00 coverage | 0.0812 area |
| off-grid, EAFI profile | 40 | 1.00 coverage | 0.0829 area |
| 10% near structural mismatch, ordinary | 40 | 1.00 acceptance | best loss 0.8641 |
| 10% near structural mismatch, EAFI | 40 | 1.00 acceptance | best loss 0.8641 |

The machine-readable result is
`P4/results/p4_eafi_robustness.json`.

## Decision

The off-grid result is encouraging for profile coverage in this controlled
family. The near-mismatch result is a required negative finding: the current
loss-normalized score does not distinguish a 10% contaminated signal from an
acceptable Mittag-Leffler observation. Increasing the error inflation factor
would not solve this problem; it would make acceptance even more permissive.

Therefore the current P4 route remains **methodologically viable but not yet
validated as a structural-mismatch detector**. The next experiment must add a
separate model-discrepancy statistic or a calibrated likelihood-ratio test,
and must include a ladder of mismatch strengths with an independently frozen
decision rule. Until that test succeeds, P4 should claim error-aware profile
inference only, not reliable automatic model rejection.

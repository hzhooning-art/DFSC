# Stage 73: PVA Group-Composition Sensitivity

## Question and frozen design

Stage 72 evaluated the first six of nine public PVA stress-relaxation curves.
This audit asks whether that result depended on that particular composition.
It enumerates all `C(9, 6) = 84` six-curve subsets and replays the unchanged
Stage 72 adapter and frozen Stage 69 certificates. No threshold, noise bin, or
decision rule was tuned on these outcomes.

## Result

All 84 subsets entered the declared scope. Every subset was classified in the
0.005 white-noise cell, passed all five rank-two evidence checks, and returned
`EVIDENCE_AGAINST_RANK_1`. The rank-one-to-rank-two criterion improvement
ranged from 491.2603 to 596.2218, with median 534.2859. The original first-six
result is therefore not a fragile consequence of that single group choice.

| Measure | Result |
|---|---:|
| Enumerated six-of-nine subsets | 84 |
| Scope eligible | 84 |
| Evidence against rank one | 84 |
| All five rank-two checks passed | 84 |
| Criterion improvement, min / median / max | 491.2603 / 534.2859 / 596.2218 |

## Claim boundary

The 84 subsets overlap heavily and come from one nine-curve dataset. They are
not 84 independent experiments, do not increase the external-task count, and
do not replace prospective confirmation. This result supports robustness to
PVA group composition only. Complete subset records are stored in
`results/pva_group_composition_sensitivity.json`.

# Stage 68: Common-Budget Model-Order Detection

Updated: 2026-09-03

## Objective

Compare information-criterion, stability, and selective model-order decisions
on exactly the same grouped multichannel observations.  Report false order
elevation, false order reduction, abstention, selective risk, total accuracy,
and runtime without removing refused trials from the main accounting.

## Frozen design

- True ranks: one and two.
- Six grouped channels.
- Horizons: 4, 8, and 16.
- Samples per channel: 24 and 48.
- Noise standard deviations: 0.001 and 0.005.
- Noise models: independent Gaussian and AR(1) with rho 0.65.
- Rank-two log-rate gaps: 0.08 and 0.32.
- Twelve deterministic seeds per cell.
- Total: 864 common-budget trials.

The compared routes are matrix-pencil AIC, AICc, BIC, BIC with a 10-point
promotion margin, cross-pencil stability, and the complete selective detector.
The selective detector uses symmetric fixed BIC evidence margins: rank one is
reported only below -10, rank two only above +10 and after the separation,
local-information, and cross-pencil checks pass; intermediate or conflicting
evidence is refused.

## Aggregate result

| Method | Coverage | Overall accuracy | Selective accuracy | False elevation | False reduction | Rank-two detection |
|---|---:|---:|---:|---:|---:|---:|
| Matrix-pencil AIC | 1.0000 | 0.6458 | 0.6458 | 0.0174 | 0.5226 | 0.4774 |
| Matrix-pencil AICc | 1.0000 | 0.6458 | 0.6458 | 0.0139 | 0.5243 | 0.4757 |
| Matrix-pencil BIC | 1.0000 | 0.6412 | 0.6412 | 0.0000 | 0.5382 | 0.4618 |
| Matrix-pencil strong BIC | 1.0000 | 0.6366 | 0.6366 | 0.0000 | 0.5451 | 0.4549 |
| Cross-pencil stability | 0.8426 | 0.4792 | 0.5687 | 0.0000 | 0.5451 | 0.2188 |
| Selective detector | 0.8356 | 0.4780 | 0.5720 | 0.0000 | 0.5365 | 0.2188 |

## Decision

The route fails.  The selective detector eliminates observed false order
elevation but does so with poor rank-two power and does not achieve a favorable
risk--coverage tradeoff against ordinary information criteria.  In particular,
an unresolved rank-two signal is still frequently promoted to a rank-one
mechanism claim.  Cross-pencil stability and the complete detector are nearly
identical on this design, so the current additional gates do not supply the
required independent decision value.

This negative result must not be rewritten as evidence of superior model-order
detection.  It shows that a gate which protects only rank-two promotion is not
enough: a rank-one claim also needs evidence that the observation design had
adequate power to reveal a scientifically relevant second rate.

## Next confirmatory design

Stage 69 must use disjoint, unopened seeds.  It will permit a rank-one decision
only when a precomputed design-power certificate shows high probability of
detecting the minimum relevant rate gap and the observed rank-two evidence is
absent.  Designs without that power return `INDETERMINATE`.  Calibration and
evaluation records must remain disjoint, and the Stage 68 records are closed to
threshold selection.

The success criterion is a prespecified risk--coverage improvement over BIC and
AICc under both white and AR(1) noise, not merely a lower error rate obtained by
refusing most trials.

## Reproduction

```console
python P5/experiments/probe_common_budget_order_detection.py
python -m unittest P5.tests.test_matrix_pencil_resolution P5.tests.test_common_budget_order_detection -v
```

The aggregate record is `P5/results/common_budget_order_detection.json`.

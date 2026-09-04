# Stage 70: Power-Certificate Risk--Coverage Sensitivity

Updated: 2026-09-03

## Objective

Audit whether the positive Stage 69 result depends on an isolated choice of
the 0.70 power-confidence lower bound. The Stage 69 calibration and evaluation
seeds remain frozen. This analysis reports a sensitivity curve; it does not
select a replacement operating point after viewing confirmation outcomes.

The specificity lower bound remains fixed at 0.90. The audited power lower
bounds are 0, 0.30, 0.50, 0.70, 0.80, and 0.90. Rank-two evidence uses the
unchanged Stage 69 detector. Only permission to issue a rank-one conclusion
changes across the curve.

## Overall curve

| Power lower bound | Qualified designs | Coverage | Selective risk | False reduction |
|---:|---:|---:|---:|---:|
| 0.00 | 24/24 | 0.8316 | 0.4280 | 0.5339 |
| 0.30 | 10/24 | 0.3472 | 0.1900 | 0.0990 |
| 0.50 | 10/24 | 0.3472 | 0.1900 | 0.0990 |
| **0.70 (frozen)** | **8/24** | **0.2943** | **0.1386** | **0.0612** |
| 0.80 | 8/24 | 0.2943 | 0.1386 | 0.0612 |
| 0.90 | 8/24 | 0.2943 | 0.1386 | 0.0612 |

The curve has three empirical plateaus rather than a single favorable point.
The frozen 0.70 operating point lies on the non-dominated risk--coverage
frontier, and thresholds from 0.70 through 0.90 produce identical decisions.
Relaxing the bound to 0.30 increases coverage by 5.30 percentage points but
also increases selective risk by 5.14 points and false reduction by 3.78
points. Removing the power requirement recovers high coverage but reproduces
the Stage 68 failure.

At the frozen point, white-noise coverage/risk are 0.2326/0.2015 and AR(1)
coverage/risk are 0.3559/0.0976. The direction of the power-certificate benefit
is therefore present in both noise families, although the achievable coverage
differs materially.

## Interpretation

This audit strengthens the claim that the certificate exposes a genuine
reliability--coverage tradeoff. It does not establish that 0.70 is universally
optimal, and equal decisions over 0.70--0.90 reflect the discrete set of 24
tested acquisition designs. A denser acquisition grid and external measured
signals are required before making continuous design recommendations.

The next comparator stage should add distinct subspace/order-selection
estimators under the same observation budget. Predictive Prony and NNLS results
already elsewhere in P5 cannot be counted as model-order comparators unless
they issue rank decisions on these same records.

## Reproduction

```console
python P5/experiments/probe_power_certificate_risk_coverage.py --stdout-summary
python -m unittest P5.tests.test_power_certificate_risk_coverage -v
```

The aggregate artifact is
`P5/results/power_certificate_risk_coverage.json`.

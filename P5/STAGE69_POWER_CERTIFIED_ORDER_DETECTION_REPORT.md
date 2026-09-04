# Stage 69: Power-Certified Selective Order Detection

Updated: 2026-09-03

## Objective

Test the Stage 68 diagnosis that a rank-one mechanism claim is defensible only
when the observation design had independently demonstrated power to reveal a
scientifically relevant second rate. Calibration and evaluation use disjoint
seed ranges. Designs without a certificate return `INDETERMINATE` instead of
turning missing rank-two evidence into a rank-one claim.

## Protocol correction retained in the record

The first instrumentation dry run used 32 calibration repeats. Even 32/32
successes have a 95% Wilson lower bound of only 0.8928, so the frozen 0.90
specificity threshold was mathematically unattainable. That run was invalid as
a qualification experiment. Its calibration and evaluation seed ranges
(offsets 1,000,000 and 2,000,000) were retired in full.

The corrected protocol uses 64 calibration repeats per rank and design, new
calibration and evaluation offsets (3,000,000 and 4,000,000), and records the
discarded dry run in the JSON artifact. This correction changes sample size
only to make the original confidence-bound criterion attainable.

## Frozen corrected design

- 24 observation designs: 3 horizons, 2 sample counts, 2 noise levels, and
  white or AR(1) noise.
- Minimum relevant second-rate gap: 0.32.
- 64 calibration trials for rank one and 64 for the relevant rank-two
  alternative in every design.
- A design qualifies only if the 95% Wilson lower bounds are at least 0.70 for
  rank-two power and 0.90 for rank-one specificity.
- 16 unopened evaluation repeats for rank one and for each rank-two gap (0.08
  and 0.32), giving 1,152 evaluation trials.
- A rank-two decision still requires strong BIC, admissibility, separation,
  local information, and cross-pencil stability. A rank-one decision additionally
  requires a qualified design certificate. All other outcomes are refused.

The prespecified success gate requires, separately under white and AR(1) noise:
coverage of at least 0.20, selective-risk improvement over AICc of at least
0.10, and false order elevation no greater than 0.05.

## Qualification and aggregate result

Eight of 24 observation designs qualified for rank-one claims. Qualification
was concentrated at the 0.001 noise level with horizons 8 or 16; only one
0.005-noise design qualified. This is an informative acquisition boundary, not
a claim that the other 16 designs contain one-rate dynamics.

| Method | Coverage | Overall accuracy | Selective accuracy | Selective risk | False elevation | False reduction | Rank-two detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matrix-pencil AICc | 1.0000 | 0.6441 | 0.6441 | 0.3559 | 0.0000 | 0.5339 | 0.4661 |
| Stage 68 selective | 0.8316 | 0.4757 | 0.5720 | 0.4280 | 0.0000 | 0.5339 | 0.2135 |
| Power-certified selective | 0.2943 | 0.2535 | 0.8614 | 0.1386 | 0.0000 | 0.0612 | 0.2135 |

The corrected route passed all prespecified checks in both noise families:

| Noise | Coverage | Candidate risk | AICc risk | Risk improvement | False elevation |
|---|---:|---:|---:|---:|---:|
| AR(1) | 0.3559 | 0.0976 | 0.2500 | 0.1524 | 0.0000 |
| White | 0.2326 | 0.2015 | 0.4618 | 0.2603 | 0.0000 |

## Interpretation and boundary

Stage 69 repairs the specific logical error exposed in Stage 68: unresolved
rank-two cases are no longer routinely converted into rank-one claims. The
price is explicit and substantial abstention (70.57% overall). Therefore the
positive result is a calibrated reliability/coverage tradeoff, not superior
unconditional classification accuracy.

The power certificate is relative to a declared 0.32 rate gap, six grouped
channels, the tested acquisition grid, and the controlled exponential
generator. It does not prove that an unobserved second rate is absent, does not
cover smaller scientifically important gaps, and is not yet evidence of
transfer to measured signals.

## Signal Processing continuation gate

Before manuscript promotion, compare the frozen detector with additional
signal-processing order estimators under the same budget, add calibration
curves rather than one operating point, and validate the certificate on held-out
physical/public signals. The main paper must show both the 29.43% coverage and
the 13.86% selective risk; reporting only selective accuracy would be
misleading.

## Reproduction

```console
python P5/experiments/probe_power_certified_order_detection.py
python -m unittest P5.tests.test_power_certified_order_detection -v
```

The aggregate artifact is `P5/results/power_certified_order_detection.json`.

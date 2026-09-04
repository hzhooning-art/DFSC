# Stage 72: External Power-Certificate Scope Transfer

Updated: 2026-09-03

## Objective

Replay the frozen Stage 69 binary order rule on existing public PVA,
KupferDigital, UCI gas-sensor, and UCI hydraulic data without modifying its
evidence thresholds. All four data sources predate Stage 69, so this is a
retrospective external transfer audit, not unopened confirmation.

## Frozen adapter and scope gate

- Exactly six source curves per evaluated group.
- At least 24 genuinely observed samples per source curve; interpolation never
  upgrades a shorter record.
- Uniform registration to 24 points over a dimensionless horizon of 16.
- Endpoint normalization maps each candidate decay from one to zero.
- Noise is mapped upward to the frozen 0.001 or 0.005 Stage 69 bin; estimates
  above 0.005 are refused.
- Residual lag-one correlation selects the frozen white/AR(1) certificate.
- Groups with more than 25% increasing increments are outside the calibrated
  monotone-decay morphology and are refused.

These transformations standardize representation but do not create independent
observations or make an excluded morphology eligible.

## Result

| Public task | Evaluated groups | Scope eligible | Frozen-rule result |
|---|---:|---:|---|
| PVA relaxation | 1 | 1 | 1 evidence against rank one |
| KupferDigital relaxation | 2 | 0 | 2 scope refusals |
| UCI gas recovery | 50 | 0 | 50 scope refusals |
| UCI hydraulic transients | 3 | 0 | 3 scope refusals |
| **Total** | **56** | **1** | **55 scope refusals** |

All 50 gas groups have only ten genuinely observed time points and therefore
fail the sample-budget gate. Both KupferDigital blocks and all three hydraulic
groups fail the monotone-decay morphology gate.

The eligible PVA group has a noise proxy of 0.002972 and maps conservatively to
the frozen 0.005 white-noise design. Its rank-one-to-rank-two criterion
improvement is 573.155, and all five rank-two checks pass: strong BIC,
admissibility, separation, local information, and cross-pencil stability. The
result is labelled `EVIDENCE_AGAINST_RANK_1`, not “supported rank two”, because
the binary audit does not exclude rank three or higher external structure.

## Decision

Stage 72 supplies one externally consistent rejection of rank-one sufficiency,
but the transfer route is not broadly successful: only 1/56 groups enters the
calibrated scope. This sharply narrows the Signal Processing claim. The method
can currently be presented as a scope-aware selective detector whose refusal
logic survives heterogeneous public data, not as a generally applicable public-
signal order selector.

The PVA result also uses one deterministic first-six-curve block. A group-
composition sensitivity audit is required before treating it as robust external
evidence. Truly prospective external confirmation remains outstanding.

## Reproduction

```console
python P5/experiments/probe_external_power_certificate_transfer.py --stdout-summary
python -m unittest P5.tests.test_external_power_certificate_transfer -v
```

The complete group-level artifact is
`P5/results/external_power_certificate_transfer.json`.

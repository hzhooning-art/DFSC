# Stage 76: Cable-Ageing Window Sensitivity

Updated: 2026-09-03

## Design

After the Stage 75 outcome was known, a sensitivity contract froze 12
start/end-window combinations: start fractions 0, 0.01, and 0.05 crossed with
end fractions 0.80, 0.90, 0.95, and 1.00. Every window had to retain at least
2,000 raw observations per curve and was passed through the unchanged
six-channel, 24-point Stage 72/75 adapter and Stage 69 decision rule.

The contract SHA-256 is
`da10666da113fed08a0da5a651e4796bea15e5946b61ff60aadab30b43ad5bc0`;
the runner SHA-256 frozen before execution is
`6f355451c8c668ac0944245ea43b19932ca7c53c26e223f627365d122cb457e7`.

## Result

All 12 windows retained at least 2,177 raw points per curve, entered the
calibrated scope, passed all five rank-two checks, and returned
`EVIDENCE_AGAINST_RANK_1`, matching the full-window Stage 75 decision. The
rank-one-to-rank-two criterion improvement ranged from 399.989 to 714.271.
Thus, the reported cable-ageing conclusion is not determined by retaining the
first 1--5% or the final 5--20% of the recorded horizon.

## Evidence boundary

This is a post-result robustness analysis, not a preregistered confirmatory
test. The 12 windows overlap heavily and share the same six source curves;
they are sensitivity settings, not 12 independent datasets, trials, or
prospective replications. The independently acquired prospective-validation
gap therefore remains open.

## Reproduction

```console
python P5/experiments/probe_cable_window_sensitivity.py
python -m unittest P5.tests.test_cable_window_sensitivity -v
```

The complete record is `P5/results/cable_window_sensitivity.json`.

Both manuscript languages compile successfully with Tectonic 0.17.0. After a
redundancy-only editorial compression, the English single-column double-spaced
build is 29 pages, leaving one page below the 30-page limit; the Chinese build
is 24 pages.

# Stage 75: Preregistered New-to-P5 Cable-Ageing Transfer

Updated: 2026-09-03

## Design

Before running the P5 decision code, the source workbook, six named curves,
worksheet coordinates, minimum raw sample count, and unchanged Stage 69/72
scope and decision thresholds were frozen in
`experiments/stage75_cable_ageing_transfer_contract.json`. The contract SHA-256
is `4a72cd0480a2cf67a5915327a77aaa77818e531a7356078b7281f1aee7d18c33`;
the first-run runner SHA-256 is
`71dcff162d2fce0b708a8864d2fd049dd747c4448abe940b427e28cc877f6e05`.

The public CC-BY-4.0 dataset contains stress-relaxation curves for unaged and
compression-aged hard, soft, and semiconductive silicone rubber used in cable
accessories (Zenodo DOI 10.5281/zenodo.18507412). Each frozen curve contains
2,757 or 2,758 finite, unique observations. The P5 adapter then uses exactly
the same six-channel, 24-sample, dimensionless-horizon-16 representation and
scope gate as Stage 72.

## Result

The six-curve group entered scope without threshold retuning. Its noise proxy
was 0.001810, conservatively mapped to the frozen 0.005 white-noise cell; the
monotonicity-violation fraction was zero. The rank-one-to-rank-two criterion
improvement was 647.951, all five rank-two checks passed, and the frozen rule
returned `EVIDENCE_AGAINST_RANK_1`. The associated design certificate was not
qualified for a positive rank-one claim, so the result is evidence against
rank-one sufficiency rather than a claim that rank two is the true physical
mechanism.

## Evidence boundary

This is the first frozen-rule application of this dataset within P5, and the
P5 contract and runner were hashed before the P5 outcome was revealed. The raw
dataset had, however, already been analysed for a different question in P3.
Consequently this is a preregistered new-to-P5 external transfer, not a newly
acquired prospective experiment or investigator-blind confirmation. It adds
an independent public dataset and a non-PVA eligible result without closing
the strongest prospective-validation gap.

## Reproduction

```console
python P5/experiments/probe_preregistered_cable_ageing_transfer.py
python -m unittest P5.tests.test_preregistered_cable_ageing_transfer -v
```

The complete record is
`P5/results/preregistered_cable_ageing_transfer.json`.

Both updated manuscript languages compile successfully with Tectonic 0.17.0.
The English single-column double-spaced build is exactly 30 pages and therefore
passes the journal limit without additional page margin; the Chinese build is
24 pages.

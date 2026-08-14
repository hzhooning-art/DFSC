# Forced MLSL Calibration Result

## Broad scan

The calibration scanned 320 combinations of fractional order, query time,
quadrature points, and Mittag-Leffler terms. The provisional trusted envelope
was `finite and max(abs(output)) <= 1e6`.

- total cases: 320;
- trusted cases: 0;
- smallest observed maximum magnitude: approximately `2.96e11`.

The machine-readable result is
`P4/results/p4_forced_backbone_calibration.json`.

## Targeted narrow probe

A separate low-mode, short-time probe found a narrow stable region:

- 1--2 retained modes were stable for `t = 0.001, 0.01, 0.05`;
- 4 modes were mostly stable at short times but reached approximately
  `3.69e3` at `alpha=0.8, t=0.05`;
- 8 modes reached approximately `3.39e6` at `alpha=0.8, t=0.01` and up to
  `2.07e17` at `t=0.05`.

## Decision

The forced primitive has a narrow numerical operating region, but not a
validated regime broad enough for the current P4 parameter-family claim. The
forcing-as-input route is therefore **temporarily blocked**, not rejected
forever. It can be reopened after P2 adds forced-propagation error control,
modal truncation checks, and an explicit abstention policy.

The current P4 mainline remains the homogeneous parameter-conditioned MLSL
family with bounded residual composition. Forcing should be reported as a
known limitation and future extension until this calibration gap is closed.

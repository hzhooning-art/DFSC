# JSS Stage 5: Cross-Project Fixed-Side Regression Replay

## Objective

Stage 5 broadens historical-defect intake beyond PyTorch without weakening the
admission rule. The installed SciPy regression suite contains an upstream case
for SciPy issue #8906, in which `solve_banded` mishandled a 1x1 system when the
band-row position was not the previously special-cased value. The replay uses
the current installed SciPy and records both the ordinary and padded-band
representations.

## Result

SciPy 1.18.0 returned the exact expected solution for both representations
(maximum absolute error 0.0) and left the right-hand side unchanged. Together
with the two existing PyTorch current-version replays, all three fixed-side
cases execute and pass across two independent SUT projects.

| Measure | Result |
|---|---:|
| Independent SUT projects with fixed-side replay | 2 |
| Fixed-side cases executed / confirmed | 3 / 3 |
| Buggy roles confirmed | 0 |
| Complete buggy/fixed pairs | 0 |

The installed SciPy regression source is provenance-fixed by SHA-256
`d40ba24f428b409d9dc8787b33a14927268ebc6e94209560f0cd696484636eee`.
Machine-readable observations are in
`results/p4_cross_project_fixed_regressions.json`.

## Claim boundary

This is a real improvement in fixed-side project diversity, but it is not yet
historical-defect detection evidence. No reported buggy SciPy environment was
executed, the complete-pair count remains zero, and the SciPy replay does not
show how the layered P4 strategy would classify the defect. Stage 5 therefore
reduces a single-project provenance weakness while leaving the main JSS
external-validity gate open.

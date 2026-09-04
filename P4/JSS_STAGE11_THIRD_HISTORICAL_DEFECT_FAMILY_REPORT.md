# JSS Stage 11: Third Complete Historical Defect Family

Date: 2026-09-04

## Outcome

SciPy issue #15620 is reproduced as a complete buggy/fixed release pair with
one unchanged runner. The official SciPy 1.14.1 wheel silently returns all-zero
`resample_poly` output for both `int16` and `int32` inputs even though the
float64 reference is nonzero and reaches 3.0015524117881807. Installed SciPy
1.18.0 returns nonzero integer-input results identical to the float64 reference
at stored precision.

## Frozen evidence

- Upstream issue: https://github.com/scipy/scipy/issues/15620
- Fix pull request: https://github.com/scipy/scipy/pull/21686
- Fix commit: `fec5d2012691729c0b49bb26277464891ac4f189`
- Buggy environment: Python 3.10.11, NumPy 2.1.3, SciPy 1.14.1
- Fixed environment: Python 3.12.14, NumPy 2.5.0, SciPy 1.18.0
- Runner SHA-256: `0a7a72bb1fb803e4c60a4531d4376e1b04659d3fb0eae3c5fea16f9ad01cdf79`
- Record: `P4/results/p4_scipy_resample_poly_pair.json`

Both role checks exit successfully. Wheel hashes, commands, versions, complete
outputs, and error metrics are retained in the JSON record.

## Claim boundary

This raises P4 from two to three complete real historical defect families and
adds a silent dtype-dependent numerical failure to the earlier wrong-gradient
and exception cases. It does not add a third independent project: the three
families still span PyTorch and SciPy only, so the manuscript does not claim
field prevalence or comprehensive defect coverage.

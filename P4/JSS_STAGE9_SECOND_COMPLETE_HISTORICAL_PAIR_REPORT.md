# JSS Stage 9: Second Complete Historical Pair Across Projects

Updated: 2026-09-03

## Design

Stage 9 executes one unchanged, dependency-minimal runner against a reported
buggy SciPy release and the current fixed release. The case is upstream issue
SciPy #8906: for a 1x1 banded system, SciPy 1.14.1 indexed row 1 even when the
conventional `(0, 0)` band representation contains only row 0. A mathematically
equivalent padded `(1, 1)` representation masks that error.

The runner SHA-256 is
`ce901b112325640d58db45dd31b1244f32e5752de85e1bbf5e3d69454c9e1bc8`.
The buggy environment uses Python 3.10.11, NumPy 2.1.3, and the official SciPy
1.14.1 Windows wheel. Its SciPy wheel SHA-256 is
`a49f6ed96f83966f576b33a44257d869756df6cf1ef4934f59dd58b25e0327e5`.
The fixed environment uses SciPy 1.18.0. Official tagged source confirms the
hard-coded row in v1.14.1 and the corrected upper-band index in v1.15.0; the
upstream fix commit is `2d20569f42a0ec1d20ce6b396c12a2b636bd15f4`.

## Result

In SciPy 1.14.1, the conventional representation raises the reported
`IndexError`, while the padded representation returns the exact solution
`[0.5, 1.0, 1.5]`. In SciPy 1.18.0, both representations return that exact
solution. Both declared environment-role checks pass.

Together with PyTorch #80770 from Stage 8, P4 now contains two complete
historical buggy/fixed environment pairs from two independently maintained
projects:

| Defect | Buggy release | Fixed-side release | Complete |
|---|---:|---:|---:|
| PyTorch #80770 | 1.11.0+cpu | 2.11.0+cu128 | yes |
| SciPy #8906 | 1.14.1 | 1.18.0 | yes |

## Evidence boundary

This closes the earlier single-project and surrogate-only limitation for the
SciPy case. It does not estimate field prevalence: the evidence still covers
only two defect families, and #8906 is a narrow 1x1 representation asymmetry.
Repeated right-hand sides are within-family trials rather than independent
historical defects.

## Reproduction

```console
python P4/experiments/p4_scipy_complete_pair_capture.py
python -m unittest P4.tests.test_scipy_complete_pair_capture -v
```

The complete commands, outputs, versions, hashes, and role decisions are in
`P4/results/p4_scipy_complete_pair.json`.

# Contributing to dfsc

Contributions should preserve the distinction between validated components,
experimental wrappers, and planned features.

1. Open an issue describing the numerical problem, operator assumptions, and expected behavior.
2. Add focused unit tests for value, gradient, shape, dtype, and device behavior where applicable.
3. Run `python -m unittest discover -s tests` and `python tools/mlsl_doctor.py`.
4. For numerical changes, include a reference solution and record tolerances rather than only plotting results.
5. For external data, add a manifest containing the source, license, citation, checksum, split, and preprocessing steps.

Bug reports should include the dfsc version, Python/PyTorch versions, device,
dtype, minimal reproducer, and reliability diagnostics.

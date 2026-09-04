# P4 JSS Stage 8: Complete Historical Buggy/Fixed Pair

Updated: 2026-09-03

## Objective

Execute the unchanged Stage-4 runner in an upstream-reported buggy package and
the current fixed environment, with immutable runtime artifacts and explicit
role checks. This closes one complete historical pair without substituting an
injected or source-derived surrogate for the old package.

## Runtime evidence

- Case: PyTorch #80770, incorrect gradient of `xlogy(0, 2)`.
- Frozen runner SHA-256:
  `fdb828ccad16c301981a9334352ae47a337b3a00ca8913e40f4b028334ee4014`.
- Buggy runtime: official PyTorch `1.11.0+cpu` wheel under the official Python
  3.10.11 Windows embedded runtime.
- Fixed runtime: PyTorch `2.11.0+cu128` under Python 3.12.13, evaluated on CPU.
- Official legacy wheel SHA-256:
  `bd984fa8676b2f7c9611b40af3a7c168fb90be3e29028219f822696bb357f472`.
- Python archive SHA-256:
  `608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d`.
- Runtime sources: Python 3.10.11 release page
  `https://www.python.org/downloads/release/python-31011/` and PyTorch's
  official wheel index `https://download.pytorch.org/whl/torch_stable.html`.

## Result

The PyTorch 1.11.0 side returned a derivative of 0 for `xlogy(0, 2)`, matching
the reported defect and differing from the analytic value
`log(2) = 0.6931471805599453`. The current side returned the analytic value at
stored precision, with zero absolute error. Both role checks passed, both
process exit codes were zero, and the runner hash was identical.

| Historical cases | Buggy roles confirmed | Fixed roles confirmed | Complete pairs |
|---:|---:|---:|---:|
| 3 | 1 | 3 | 1 |

## Claim boundary

This is one complete upstream defect family in one software project. It
substantially improves historical realism but does not estimate defect
prevalence, prove broad field effectiveness, or replace replication across
additional defects and projects. PyTorch #30303 remains fixed-side-only because
its buggy reproduction requires a compatible legacy CUDA runtime.

The machine-readable record is
`results/p4_complete_historical_pair.json`.

Both updated manuscript languages compile successfully with Tectonic 0.17.0;
the English and Chinese audit builds contain 34 and 28 pages, respectively.

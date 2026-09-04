# P4 JSS Stage 4: Cross-Version Historical Replay Harness

Updated: 2026-09-03

## Objective

Freeze one minimal runner that can be executed unchanged in reported buggy and
current fixed PyTorch environments. This prevents the two sides of a historical
pair from silently using different tests. The runner depends only on the Python
standard library and PyTorch and emits a versioned JSON decision.

## Frozen cases

- PyTorch #80770: `xlogy(0, 2)` should have derivative `log(2)`; the reported
  PyTorch 1.11.0 behavior returned zero.
- PyTorch #30303: CUDA `einsum` on a non-contiguous expanded input should agree
  with matmul; the reported PyTorch 1.3.1 behavior produced a wrong gradient.

The runner separately records `matches_reported_bug` and
`matches_fixed_expectation`. The expected environment role is supplied on the
command line, and a role is confirmed only by the corresponding numerical
condition.

## Current fixed-side result

Both cases executed under PyTorch 2.11.0+cu128 and confirmed the fixed role.
The `xlogy` derivative error was zero at stored precision, and the CUDA
matmul--einsum gradient disagreement was zero. The frozen runner SHA-256 is:

`fdb828ccad16c301981a9334352ae47a337b3a00ca8913e40f4b028334ee4014`

| Attempted | Executed | Fixed roles confirmed | Buggy roles confirmed | Complete pairs |
|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 0 | 0 |

## Blocking environment audit

The host exposes no additional Python installation, no Docker executable, and
no cached PyTorch/JAX/PyBNF legacy wheel. Consequently, neither PyTorch 1.11.0
nor 1.3.1 can be executed locally without obtaining a compatible isolated
runtime. This limitation is recorded rather than replacing the old library
with an emulated bug.

Stage 4 completes the replay harness and fixed-side evidence only. It does not
change the JSS historical-defect count. The pair becomes admissible only when
the identical runner hash confirms the reported failure in the old environment.

**Later update (Stage 8, 2026-09-03):** the official PyTorch 1.11.0+cpu wheel
was obtained and the unchanged runner confirmed the #80770 buggy role. Together
with the current fixed-side run, this now forms one complete pair. See
`JSS_STAGE8_COMPLETE_HISTORICAL_PAIR_REPORT.md`; the statements above describe
the Stage-4 state at the time of that audit.

## Reproduction

```console
python P4/experiments/pytorch_historical_pair_runner.py --case pytorch_80770 --expected-role fixed
python P4/experiments/pytorch_historical_pair_runner.py --case pytorch_30303 --expected-role fixed
python P4/experiments/p4_historical_fixed_replays.py --stdout-summary
```

The execution matrix is `P4/docs/P4_HISTORICAL_REPLAY_MATRIX.md`; the fixed-side
artifact is `P4/results/p4_historical_fixed_replays.json`.

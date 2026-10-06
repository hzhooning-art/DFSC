# P4 plan execution: first verified vertical slice

Updated: 2026-09-30

The `p4-testing execute` command runs every relation selected by a risk-aware
plan through an explicitly supplied local command registry. It checks that the
plan still recompiles from the supplied contract, then stores each worker's
raw stdout/stderr, process status, timing, command and file hashes, reported
subject-call count, and the final tri-state result. An absent worker, timeout,
process error, malformed observation, zero-call `PASS`, or non-evaluable result
cannot yield `CONFORMANT`. A valid observed `FAIL` yields `NONCONFORMANT` even
if another relation cannot be evaluated. The total wall-time budget is enforced
across the selected subprocesses.

The command registry is **trusted operator configuration**. It can execute
arbitrary programs and must not be accepted from untrusted contracts or public
submissions. Commands are passed as argv with `shell=False`. Supported
substitutions are `{python}`, `{p4_root}`, and `{test_id}`. The worker must emit
one JSON object with schema `P4-Relation-Observation-v1`, the exact selected
`test_id`, an outcome of `PASS`, `FAIL`, or `NON_EVALUABLE`, an integer
`subject_invocations`, and a `details` object. Worker oracles and reported call
counts remain the worker author's responsibility; command hashes are audit
links, not authentication or proof of correct oracle construction.

The dependency-free `math.expm1` example executes a value check against
high-precision `decimal` and a centered finite-difference derivative check.
It is an engineering smoke test of the execution architecture, **not** a new
historical-defect family or evidence of field-wide effectiveness.

From the repository root on Windows (with a working Python 3.10+):

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
& 'D:\python\python.exe' -m p4_testing.cli compile `
  --contract P4/contracts/examples/expm1_execution_demo.json `
  --risk-profile P4/contracts/examples/expm1_execution_risk.json `
  --seed 17 --max-invocations 2 --wall-seconds 5 --output P4/results/expm1_plan_demo.json
& 'D:\python\python.exe' -m p4_testing.cli execute `
  --contract P4/contracts/examples/expm1_execution_demo.json `
  --plan P4/results/expm1_plan_demo.json `
  --registry P4/contracts/examples/expm1_execution_registry.json `
  --output P4/results/expm1_execution_demo.json
```

The planned `max_invocations` value still means *relation slots* in the
existing compiler, not observed subject calls. The execution report separately
records worker-reported calls; the example selects two relations and reports
six subject calls. No equal-cost comparison should use the planning number as
runtime cost.

Stage 2 now executes the selected value-reference relation in the existing
SciPy #4130 contract against an installed production wheel; see
`SCIPY_4130_LIVE_REFERENCE_EXECUTION.md`. Stage 3 also executes its selected
historical-replay relation with the unchanged frozen runner in the 1.8.1 and
1.16.3 wheels; see `SCIPY_4130_FULL_PLAN_EXECUTION.md`. Next gate: independently
verify production-component workers and full selected plans across the other
frozen cases. One complete case does not establish broad effectiveness.

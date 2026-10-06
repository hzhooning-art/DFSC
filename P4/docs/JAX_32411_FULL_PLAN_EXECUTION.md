# JAX #32411: first cross-project live selected-relation plan

Updated: 2026-10-02

This stage executes both relations selected from the existing JAX #32411
component contract. The frozen, role-independent runner is executed under
JAX 0.7.2 and 0.8.0 for `replay.version_pair`, then under JAX 0.8.0 again
for `property.gradient_accumulation_equivalence`. The worker validates the
frozen runner SHA-256, versions, tensor shape, microbatch count, outcome, and
concrete gradient/value errors; it does not accept the outcome string alone.

| Selected relation | New observation | Outcome |
| --- | --- | --- |
| `replay.version_pair` | JAX 0.7.2 gradient error `0.03254644572734833`; JAX 0.8.0 error `0.0` | `PASS` |
| `property.gradient_accumulation_equivalence` | Current JAX 0.8.0 ref-accumulation and value-accumulation paths agree with zero gradient/value error | `PASS` |

The [compiled plan](../results/reconstruction/jax_32411_full_execution_plan.json)
and [raw execution report](../results/reconstruction/jax_32411_full_execution_072_vs_080.json)
show `CONFORMANT` for two selected relations and six worker-reported subject
invocations. Four belong to the old/new replay and two to the repeated
fixed-wheel property check. The child stdout/stderr, process commands,
interpreter and runner hashes, contract/registry hashes, and plan digest are
retained. The older `jax_32411_pair.json` capture was not used as a new
observation.

To repeat locally from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$planPath = Join-Path $env:TEMP 'jax_32411_full_plan.json'
$reportPath = Join-Path $env:TEMP 'jax_32411_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/jax_32411.json `
  --risk-profile P4/contracts/cases/jax_32411_full_risk.json `
  --seed 32411 --max-invocations 2 --wall-seconds 30 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/jax_32411.json `
  --plan $planPath `
  --registry P4/contracts/cases/jax_32411_full_registry.json `
  --output $reportPath
```

The registry expects ignored local environments
`P4/.runtime/p4_jax072/Scripts/python.exe` and
`P4/.runtime/p4_jax080/Scripts/python.exe`; they are not included in the
tracked source artifact. Missing environments, altered frozen runner, invalid
version/shape/error semantics, or an unreproduced old failure cause
`NON_EVALUABLE` and plan abstention. The worker deliberately removes an
inherited `KMP_DUPLICATE_LIB_OK` override from child environments. No unsafe
OpenMP workaround is used. The contract uses JAX's private `vjp3` API, so
portability to later JAX versions is not implied.

The current property relation reuses the same frozen comparison algorithm as
the fixed side of the replay; it is a fresh process execution, **not** an
independent oracle or independent scientific replication. This result tests
one already admitted JAX family and the executor's second-project
applicability. It does not add a new defect family or establish broad
cross-project effectiveness.

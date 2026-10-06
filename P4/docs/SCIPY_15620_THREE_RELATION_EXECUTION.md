# SciPy #15620: three-layer live plan execution

Updated: 2026-10-02

The existing SciPy #15620 contract now executes all three selected relations:

| Relation | Work performed | Outcome |
| --- | --- | --- |
| `replay.version_pair` | Run the unchanged, SHA-256-bound historical runner in SciPy 1.14.1 and 1.16.3; inspect four dtype outputs per role | `PASS` |
| `reference.float64_reference` | Call current `resample_poly` once with float64 and compare against a frozen recorded vector | `PASS` |
| `property.dtype_consistency` | Call current `resample_poly` with float64, int16, and int32; compare integer outputs to float64 | `PASS` |

The old SciPy 1.14.1 wheel returned all-zero int16 and int32 outputs while
its float64 result was nonzero. SciPy 1.16.3 returned integer outputs that
match float64 within the declared tolerance. The complete plan reported
`CONFORMANT` and 12 worker-reported subject calls (8 in replay, 1 in the
reference relation, 3 in the dtype property relation). The [compiled plan](../results/reconstruction/scipy_15620_full_execution_plan.json)
and [raw report](../results/reconstruction/scipy_15620_full_execution_1141_vs_1163.json)
retain child output, commands, hashes, versions, and observations. The older
captured-pair JSON was **not** used as a new observation.

The [float64 fixture](../contracts/cases/scipy_15620_reference_fixture.json)
is version-stable across local 1.14.1 and 1.16.3 runs. It is **not** an
independent mathematical oracle: the reference and property relations both
depend on the same recorded signal response. Therefore this evidence supports
the specified cross-dtype consistency and historical repair, not absolute
correctness of `resample_poly` for arbitrary signals. A changed fixture,
missing old interpreter, changed frozen runner, or unreproduced old role
cannot silently pass. The ignored local 1.14.1 environment is not included
in the tracked source tree, so outside reproduction needs equivalent wheels.

To repeat locally from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$env:P4_SCIPY_SITE_PACKAGES = 'D:\Anaconda\Lib\site-packages'
$env:P4_SCIPY_DLL_DIR = 'D:\Anaconda\Library\bin'
$planPath = Join-Path $env:TEMP 'scipy_15620_full_plan.json'
$reportPath = Join-Path $env:TEMP 'scipy_15620_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/scipy_15620.json `
  --risk-profile P4/contracts/cases/scipy_15620_full_risk.json `
  --seed 15620 --max-invocations 3 --wall-seconds 25 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/scipy_15620.json `
  --plan $planPath `
  --registry P4/contracts/cases/scipy_15620_full_registry.json `
  --output $reportPath
```

This is live execution evidence for a **fourth existing family**. The
independent-family benchmark remains seven cases across three projects.

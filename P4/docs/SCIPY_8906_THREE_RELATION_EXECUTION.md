# SciPy #8906: three-layer live plan execution

Updated: 2026-09-30

The existing SciPy #8906 case now has a fully executed three-relation plan:

| Selected relation | Actual call(s) | Result |
| --- | --- | --- |
| `reference.reference_solution` | Current `solve_banded` on conventional singleton representation; compare to analytic `[0.5, 1, 1.5]` | `PASS` |
| `property.representation_equivalence` | Current `solve_banded` on conventional and padded representations; compare both to each other **and** to the analytic solution | `PASS` |
| `replay.version_pair` | Same SHA-256-bound frozen runner in SciPy 1.14.1 and 1.16.3 | `PASS` |

The older wheel raised `IndexError` on the conventional representation but
returned the analytic solution for the padded representation. The current
wheel returned that solution for both. The replay worker checks those
observations directly; it does not accept `role_confirmed: true` alone.
The complete plan is `CONFORMANT` with seven worker-reported subject calls:
four in the historical pair, one in the reference check, and two in the
property check. The [plan](../results/reconstruction/scipy_8906_full_execution_plan.json)
and [raw report](../results/reconstruction/scipy_8906_full_execution_1141_vs_1163.json)
retain process output, commands, file hashes, plan/contract/registry hashes,
and observed values. The prior captured-pair JSON is not used as a new
observation.

To repeat on this Windows machine from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$env:P4_SCIPY_SITE_PACKAGES = 'D:\Anaconda\Lib\site-packages'
$env:P4_SCIPY_DLL_DIR = 'D:\Anaconda\Library\bin'
$planPath = Join-Path $env:TEMP 'scipy_8906_full_plan.json'
$reportPath = Join-Path $env:TEMP 'scipy_8906_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/scipy_8906.json `
  --risk-profile P4/contracts/cases/scipy_8906_full_risk.json `
  --seed 8906 --max-invocations 3 --wall-seconds 20 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/scipy_8906.json `
  --plan $planPath `
  --registry P4/contracts/cases/scipy_8906_full_registry.json `
  --output $reportPath
```

The ignored local `P4/.runtime/p4_scipy114_replay/python310/python.exe`
environment is required for the historical side and is not included in the
tracked artifact. Missing executables, a changed frozen runner, malformed
observations, or an unreproduced historical role cause abstention. The
reported Python hashes and SciPy versions are provenance evidence, not an
archive of installed wheel bytes. Worker-reported call counts are not
independently instrumented. Both current relations use the same small
analytic oracle and are not independent scientific replications.

This strengthens the **third of seven existing historical defect families**.
It does not add a new family or establish broad effectiveness, contract
authoring reliability, or superiority over other test strategies.

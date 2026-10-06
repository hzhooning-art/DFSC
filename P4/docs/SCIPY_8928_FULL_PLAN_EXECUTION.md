# SciPy #8928: second live two-relation plan

Updated: 2026-09-30

The existing SciPy #8928 contract now has a complete selected-relation run.
The risk profile selects `replay.version_pair` and
`reference.analytic_mean`. The replay worker executes the **same frozen
runner** in a local SciPy 1.8.1 environment and a post-fix SciPy 1.16.3
environment. The separate reference worker calls `scipy.stats.norm.expect`
for the two declared location-scale inputs on the current wheel and compares
each output to the analytic normal mean `E[X] = loc`.

| Relation | Live result | Observation |
| --- | --- | --- |
| Historical pair | `PASS` | 1.8.1 failed both cases; 1.16.3 passed both |
| Current analytic reference | `PASS` | Maximum absolute error `1.4210854715202004e-14`, below `1e-10` |

The old wheel's maximum absolute error was `35.99999999966363`; the current
wheel's maximum was `1.4210854715202004e-14`. The [compiled plan](../results/reconstruction/scipy_8928_full_execution_plan.json)
and [raw execution report](../results/reconstruction/scipy_8928_full_execution_181_vs_1163.json)
record a `CONFORMANT` result for both selected relations and six
worker-reported subject calls (two old, two current in replay, two current
in the separate reference relation). The report retains raw child output,
stderr, command/file hashes, case/contract/registry hashes, and plan digest.
The earlier `scipy_8928_pair.json` capture was not used as fresh observation.

To repeat on this Windows machine from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$env:P4_SCIPY_SITE_PACKAGES = 'D:\Anaconda\Lib\site-packages'
$env:P4_SCIPY_DLL_DIR = 'D:\Anaconda\Library\bin'
$planPath = Join-Path $env:TEMP 'scipy_8928_full_plan.json'
$reportPath = Join-Path $env:TEMP 'scipy_8928_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/scipy_8928.json `
  --risk-profile P4/contracts/cases/scipy_8928_full_risk.json `
  --seed 8928 --max-invocations 2 --wall-seconds 15 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/scipy_8928.json `
  --plan $planPath `
  --registry P4/contracts/cases/scipy_8928_full_registry.json `
  --output $reportPath
```

This local replay requires the ignored
`P4/.runtime/p4_scipy181_replay/python310/python.exe` environment. A
missing interpreter, changed frozen runner, malformed observation, or
unreproduced role semantics yields `NON_EVALUABLE`/plan `ABSTAIN`, not a
success claim. The local old-wheel environment is not in the tracked artifact;
third parties must reconstruct an equivalent wheel environment. The
interpreter hashes and reported SciPy versions are recorded, but the report
does not by itself archive the installed wheel bytes. Worker-reported call
counts are not independently instrumented.

This strengthens a **second one of the same seven admitted historical defect
families**. It neither increases sample size nor supports population-wide
effectiveness or universal superiority claims. The analytic reference and
historical runner share the same mathematical oracle but use separate worker
implementations; they should not be presented as independent scientific
replications.

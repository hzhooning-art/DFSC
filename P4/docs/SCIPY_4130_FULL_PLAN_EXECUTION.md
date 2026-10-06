# SciPy #4130: two-relation live plan execution

Updated: 2026-09-30

This stage executes **both** relations selected from the existing SciPy #4130
component contract under a two-slot risk-aware plan:

| Selected relation | Actual work | Outcome |
| --- | --- | --- |
| `replay.version_pair` | Run the same SHA-256-bound historical runner in SciPy 1.8.1 and SciPy 1.16.3 | `PASS` |
| `reference.exact_enumeration` | Call the current `scipy.stats.fisher_exact` and compare with integer-hypergeometric enumeration | `PASS` |

The older wheel returned p-value `1.0`, an absolute error of
`0.04332213600735646` against the enumerated reference
`0.9566778639926435`. The post-fix SciPy 1.16.3 wheel returned
`0.9566778639926432`, an error of `3.3306690738754696e-16`. The original
runner hash matched the frozen case manifest and the exact same runner source
was executed in both roles. The full plan status was `CONFORMANT`, with two
selected relations executed and three worker-reported SciPy subject calls.

The [compiled plan](../results/reconstruction/scipy_4130_full_execution_plan.json)
and [execution report](../results/reconstruction/scipy_4130_full_execution_181_vs_1163.json)
contain the plan/contract/registry hashes, old and new interpreter hashes,
runner hash, child commands, original child stdout/stderr and their hashes,
observed values, current-site reference result, and timing. The pre-existing
`scipy_4130_pair.json` capture was **not** used as a new observation.

To repeat this local run from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$env:P4_SCIPY_SITE_PACKAGES = 'D:\Anaconda\Lib\site-packages'
$env:P4_SCIPY_DLL_DIR = 'D:\Anaconda\Library\bin'
$planPath = Join-Path $env:TEMP 'scipy_4130_full_plan.json'
$reportPath = Join-Path $env:TEMP 'scipy_4130_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/scipy_4130.json `
  --risk-profile P4/contracts/cases/scipy_4130_full_risk.json `
  --seed 4130 --max-invocations 2 --wall-seconds 15 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/scipy_4130.json `
  --plan $planPath `
  --registry P4/contracts/cases/scipy_4130_full_registry.json `
  --output $reportPath
```

The replay worker expects the ignored local
`P4/.runtime/p4_scipy181_replay/python310/python.exe` environment, which
contains the historical SciPy 1.8.1 wheel. It will return `NON_EVALUABLE` if
that interpreter is missing or if the frozen runner hash changes. The current
Python environment must supply a post-fix SciPy wheel; the local 1.16.3 path
above is a loading workaround, not part of the contract. The worker validates
that the current version is at least the contract's fixed release (1.9.0).
The ignored historical environment is **not** contained in the tracked source
tree or in this report, so third-party reproduction still needs that wheel
environment or an equivalent reconstruction with recorded hashes.

This is new execution evidence for **one already admitted defect family**.
It does not create an eighth family, measure population-wide effectiveness,
or prove that engineers can write correct contracts and independent oracles.
The plan budget counts selected relation slots, not measured subject calls;
worker-reported call counts remain subject to independent audit.

# PyTorch #80770: complete live three-relation plan

Updated: 2026-10-02

The existing `torch.xlogy` zero-input derivative contract selects an analytic
reference, exact-repeatability check, and historical version-pair replay.
All three relations were executed through the trusted local registry and
saved [compiled plan](../results/reconstruction/pytorch_80770_full_execution_plan.json)
and [raw report](../results/reconstruction/pytorch_80770_full_execution_111_vs_2110.json).
The report binds to the plan digest and is `CONFORMANT` with three `PASS`
observations and five worker-reported subject calls.

| Relation | Fresh observation | Outcome |
| --- | --- | --- |
| `replay.version_pair` | Same frozen runner: PyTorch 1.11.0+cpu returned gradient 0; 2.11.0+cu128 returned `0.6931471805599453` | `PASS` |
| `execution.repeatability` | Two independent current-wheel processes each returned `0.6931471805599453` | `PASS` |
| `reference.analytic_derivative` | Current-wheel gradient equaled analytic `log(2)` with zero recorded absolute error | `PASS` |

The worker checks the frozen runner SHA-256 from the admitted benchmark case,
the declared scalar/dtype/CPU domain, release versions, numeric errors and
role semantics. It records raw child stdout/stderr, command vectors and
interpreter hashes. The older historical capture was **not** used as a fresh
observation. Negative controls verify abstention on missing/mismatched inputs
and a `FAIL` for a wrong current gradient or inconsistent repetition.

Reproduction from the repository root in PowerShell (requires the ignored
local old interpreter and `P1/.venv` current interpreter):

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$planPath = Join-Path $env:TEMP 'p4_80770_full_plan.json'
$reportPath = Join-Path $env:TEMP 'p4_80770_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/pytorch_80770.json `
  --risk-profile P4/contracts/cases/pytorch_80770_full_risk.json `
  --seed 80770 --max-invocations 3 --wall-seconds 90 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/pytorch_80770.json --plan $planPath `
  --registry P4/contracts/cases/pytorch_80770_full_registry.json `
  --output $reportPath
```

The current wheel is CUDA-labelled but this test is CPU-only; no CUDA device
is required. The interpreters are local, ignored assets and are not packaged
with the report. The reference and repeatability relations reuse the frozen
runner algorithm, so they are distinct process executions, not independent
scientific replications. Worker-reported call counts are not independently
instrumented. This validates one already admitted historical defect family,
not a broader software-testing effectiveness rate.

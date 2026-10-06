# PyTorch #99638: complete isolated three-relation plan

Updated: 2026-10-02

The existing masked-select gradient contract selects a property comparison
against equivalent index selection, deterministic repetition, and historical
replay. The [compiled plan](../results/reconstruction/pytorch_99638_full_execution_plan.json)
and [raw execution report](../results/reconstruction/pytorch_99638_full_execution_1131_vs_2110.json)
are digest-bound. The report is `CONFORMANT`: all three relations have fresh
`PASS` observations and ten worker-reported subject invocations.

| Relation | Fresh observation | Outcome |
| --- | --- | --- |
| `replay.version_pair` | Frozen runner on PyTorch 1.13.1+cpu: maximum bias-gradient disagreement `2.816199779510498`; 2.11.0+cu128: zero | `PASS` |
| `execution.deterministic_repeat` | Two separate current-wheel processes returned identical five-element observed and reference gradient vectors | `PASS` |
| `property.selection_gradient_equivalence` | Current-wheel masked and indexed selection gradients agreed with zero maximum error | `PASS` |

Each role runs in an explicit interpreter with inherited `PYTHONPATH`,
`PYTHONHOME`, SciPy bridge variables, and `KMP_DUPLICATE_LIB_OK` removed.
The worker checks the frozen runner SHA-256, declared input shape/seed,
float32 CPU domain, versions, five-element vectors, finite errors, outcome
consistency, and historical role. Raw stdout/stderr, argv, runner and
interpreter hashes are saved. The previous captured pair is **not** treated
as a new execution. Negative controls check missing/mismatched inputs,
wrong current gradients, inconsistent repetition, and a nonreproduced old
role. No unsafe OpenMP override was used.
The previously deselected current-side candidate test now calls the isolated
current interpreter as well; all 173 reconstruction tests pass without
deselection in this local environment.

To reproduce from the repository root in PowerShell, using the ignored local
`P4/.runtime/p4_torch1131_replay/python310/python.exe` and
`P1/.venv/Scripts/python.exe` environments:

```powershell
$env:PYTHONPATH = (Resolve-Path P4/src).Path
$planPath = Join-Path $env:TEMP 'p4_99638_full_plan.json'
$reportPath = Join-Path $env:TEMP 'p4_99638_full_report.json'
& 'D:\python\python.exe' -B -m p4_testing.cli compile `
  --contract P4/contracts/cases/pytorch_99638.json `
  --risk-profile P4/contracts/cases/pytorch_99638_full_risk.json `
  --seed 99638 --max-invocations 3 --wall-seconds 90 --output $planPath
& 'D:\python\python.exe' -B -m p4_testing.cli execute `
  --contract P4/contracts/cases/pytorch_99638.json --plan $planPath `
  --registry P4/contracts/cases/pytorch_99638_full_registry.json `
  --output $reportPath
```

The current wheel is CUDA-labelled, but this case runs only on CPU. The
property and repetition checks reuse the same frozen comparison algorithm;
they are distinct processes, not independent scientific replications.
Worker call counts are not independently instrumented. The ignored local
wheel environments are not a portable bundled artifact. This result closes
execution coverage for one already admitted defect family and does not
establish performance on unseen defects or field-wide effectiveness.

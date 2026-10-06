# Selected-relation live execution coverage

Updated: 2026-10-02. The [machine-readable coverage record](../results/reconstruction/live_relation_coverage.json)
is checked against the seven admitted cases in the frozen benchmark.
"Live complete" means every relation selected in a saved plan has a fresh
execution observation and the plan-bound report has no abstention; it does
**not** mean the method is effective across the field.

| Existing family | Project | Complete live selected-relation plan | Evidence |
| --- | --- | --- | --- |
| SciPy #4130 | SciPy | Yes, 2/2 selected relations `PASS` | `scipy_4130_full_execution_181_vs_1163.json` |
| SciPy #8928 | SciPy | Yes, 2/2 `PASS` | `scipy_8928_full_execution_181_vs_1163.json` |
| SciPy #8906 | SciPy | Yes, 3/3 `PASS` | `scipy_8906_full_execution_1141_vs_1163.json` |
| SciPy #15620 | SciPy | Yes, 3/3 `PASS` | `scipy_15620_full_execution_1141_vs_1163.json` |
| JAX #32411 | JAX | Yes, 2/2 `PASS` | `jax_32411_full_execution_072_vs_080.json` |
| PyTorch #80770 | PyTorch | Yes, 3/3 `PASS` | `pytorch_80770_full_execution_111_vs_2110.json` |
| PyTorch #99638 | PyTorch | Yes, 3/3 `PASS` | `pytorch_99638_full_execution_1131_vs_2110.json` |

The resulting execution-coverage count is **7/7 admitted families**, not
seven newly discovered defects. Four complete plans come from SciPy, one
from JAX, and two from PyTorch. An old/new pair alone cannot complete a plan
whose reference or property relation remains unexecuted.

PyTorch #80770 uses an analytic `log(2)` reference, two independent current
processes for exact numeric repeatability, and an unchanged frozen runner
under old and current wheels. The current reference and repeatability checks
reuse the same runner algorithm, so they are fresh executions but not
independent methods. See [execution details](PYTORCH_80770_FULL_PLAN_EXECUTION.md).
PyTorch #99638 uses an isolated old/new CPU replay and fresh current-wheel
property and deterministic-repeat checks; see
[execution details](PYTORCH_99638_FULL_PLAN_EXECUTION.md). Both PyTorch
relations still depend on ignored local interpreters, and the seven selected
contracts were not a held-out sample. Full execution coverage is an internal
completeness milestone, not a field-effectiveness estimate.

For SciPy #15620, the float64 reference vector is a recorded numerical
fixture that agreed across local 1.14.1 and 1.16.3 runs. It is **not** an
independent mathematical oracle. The result supports dtype consistency and
historical regression recovery for the declared input, not absolute signal
processing correctness. The old 1.14.1 runtime is an ignored local asset,
not a portable tracked dependency. Reported call counts come from workers
and are not independently instrumented.

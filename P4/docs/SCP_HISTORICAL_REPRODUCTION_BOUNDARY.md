# Review-artifact historical evidence boundary

Updated: 2026-10-05. This document accompanies the SCP OSP preparation snapshot.

Start with the installed standard-library example in
`SCP_SOFTWARE_USER_GUIDE.md`. Its success, injected failure and edited-plan
refusal are portable software checks. They are not historical defect discoveries.

The historical tier is indexed by `LIVE_RELATION_COVERAGE.md` and
`results/reconstruction/live_relation_coverage.json`. The review ZIP contains
the seven cases' contracts, workers, frozen runners, saved plans/reports and
case guides. It does not contain old scientific-library wheels, local Python
environments, the current `P1/.venv`, or a hosted reproducible capsule. A saved
report is recorded developer evidence, not a newly reproduced observation.

## Runtime requirements

| Case | Recorded old / current libraries | Worker guide |
| --- | --- | --- |
| scipy_4130 | SciPy 1.8.1 / 1.16.3 | SCIPY_4130_FULL_PLAN_EXECUTION.md |
| scipy_8928 | SciPy 1.8.1 / 1.16.3 | SCIPY_8928_FULL_PLAN_EXECUTION.md |
| scipy_8906 | SciPy 1.14.1 / 1.16.3 | SCIPY_8906_THREE_RELATION_EXECUTION.md |
| scipy_15620 | SciPy 1.14.1 / 1.16.3 | SCIPY_15620_THREE_RELATION_EXECUTION.md |
| jax_32411 | JAX 0.7.2 / 0.8.0 | JAX_32411_FULL_PLAN_EXECUTION.md |
| pytorch_80770 | PyTorch 1.11.0+cpu / 2.11.0+cu128, CPU checks | PYTORCH_80770_FULL_PLAN_EXECUTION.md |
| pytorch_99638 | PyTorch 1.13.1+cpu / 2.11.0+cu128, CPU checks | PYTORCH_99638_FULL_PLAN_EXECUTION.md |

The original guides record the development-machine commands, not universally
portable commands. SciPy and JAX workers accept explicit interpreter paths;
some PyTorch paths are local worker constants. Provision equivalent environments
and review path adaptations before rerunning. Preserve the original frozen
runner bytes, record every adaptation, recompile plans when contracts change,
and pass `--artifact-root P4` when executing via the installed wheel. Missing
runtimes must remain non-evaluable, never be reinterpreted as passing tests.

## Frozen metadata clarification

The original `benchmark/frozen.json`, `FREEZE_METADATA.json`, case records and
captured results are retained byte-for-byte. They are historical provenance,
not a current dependency lockfile. In particular, the old `scipy_15620` label
`resample-poly-padtype-line` and its observable text do not accurately describe
the executed dtype regression. The current contract is
`scipy-resample-poly-dtype-v1`: integer input produces all-zero output in the
old runtime; fixed integer output agrees with float64. The upstream issue
https://github.com/scipy/scipy/issues/15620 confirms this dtype defect and its
1.15.0 milestone. The saved current execution used 1.16.3; an older manifest's
1.18.0 value is not the runtime used in that report or the first fixed release.

Use the current contract, unchanged runner, per-role version evidence and
plan-bound report together when interpreting a worked case. This explicit
clarification does not change membership, create an eighth family or rewrite
the original freeze hash. It also does not make the float64 fixture an
independent mathematical oracle. No detection-rate comparison is claimed by
the OSP manuscript.

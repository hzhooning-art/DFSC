# Full-attempt recovery — 25 October 2026

This is a separate, post-results recovery experiment, not an extension or replacement of the frozen 24 October experiment. Neither the numerical software candidate nor the manuscript is changed.

## What is running

- A: all 175 original jobs (one excluded API preflight, 78 forward, 72 gradient/path audits, 24 inverse attempts), freshly executed because the old A timings were affected by a residual CPU-consuming process.
- B: all 336 original completed coordinate records are imported byte-for-byte, including the 2 in-domain refusals and 8 outside-domain refusals. No B rerun or fresh B timing claim.
- C: all 72 original joint inverse jobs, requesting 100 updates each, freshly attempted across all sizes, regimes and seeds. Original completed/partial/refused records are retained separately, not overwritten or pooled as new independent samples.

Scientific points, seeds, tolerances, independent references and the wheel/source candidate remain unchanged. Resource limits alone increase: A 7200 seconds; C 21600 seconds; ordinary worker 600 seconds; inverse worker 900 seconds, each further limited by remaining track budget. One numerical worker at a time. Completion of the schedule is not a promise that every solver will pass or return.

The recovery copy corrects three reviewed issues: an unresolved initial-state finite difference no longer destroys a valid native gradient; standalone test freezes record missing Git provenance rather than crashing; the qualified original audit explicitly excludes old A costs from eligible performance evidence. A Windows timeout kills the precise owned worker tree instead of only the venv launcher. No third-party code is patched.

## Progress and commands

The first strict entry stopped **before any new A/C scientific job** because independent RAG/P4 services were present. They were not terminated. The active recovery route is now the explicit shared-host amendment `SHARED_HOST_AMENDMENT.json`; its entry is `run_observed_host.py`. Read **`SHARED_HOST_STATUS.json`** for its controller status. All new shared-host timings are retained as raw observations but excluded from eligible uncontended-performance comparisons. The original `RUN_STATUS.json` remains the immutable history of the stopped strict-entry attempt, not the current execution status.

```powershell
& 'P1/.venv/Scripts/python.exe' -B -X utf8 'P1/revision_neurocomputing_20261016/targeted_validation_recovery_20261025/run_observed_host.py'
# Only after inspecting a genuinely interrupted run:
& 'P1/.venv/Scripts/python.exe' -B -X utf8 'P1/revision_neurocomputing_20261016/targeted_validation_recovery_20261025/run_observed_host.py' --resume
```

The shared-host entry preserves the same frozen scientific sources and budgets, holds both experiment locks, runs A then C automatically, checks source bindings, and attempts regression checks/audit/packaging afterward. It does not overwrite completed terminal cells or reset time. `completed_with_recorded_outcomes` means execution/delivery ended with the recorded findings, not universal scientific success. A clean isolated timing study remains a separate requirement.

Read `RUN_STATUS.json` for the controller state and `results/A/ledger.json` or `results/C/ledger.json` for per-track active time and current job. Completed terminal files are immutable. Python-process checks run at track boundaries; they do not certify that the entire host has no contention. An unexpected Python process stops execution without killing unrelated work.

From the repository root:

```powershell
& 'P1/.venv/Scripts/python.exe' -B -X utf8 'P1/revision_neurocomputing_20261016/targeted_validation_recovery_20261025/recover_experiments.py' --prepare
& 'P1/.venv/Scripts/python.exe' -B -X utf8 'P1/revision_neurocomputing_20261016/targeted_validation_recovery_20261025/recover_experiments.py' --run
```

Prepare is create-only. After an interruption, inspect `RUN_STATUS.json` and use `--run --resume`; never delete finished terminal files or reset consumed budgets. Interrupted attempts remain explicit rather than being silently counted complete. An exhausted recovery budget still produces missing records and is not automatically extended.

After A and C terminate, the controller verifies old-file preservation, generates `results/audit.json` and `RESULTS_REPORT.md`, runs recovery and prior regression suites, and attempts `output/DFSC_Targeted_Validation_Recovery_20261025.zip` with manifest/hash/extraction replay. Any failure changes the status to `needs_attention`; it is not a completed delivery. `results/original_qualified_audit.json` is the authoritative qualification of the old mechanical audit; old A cost minima are retained as raw data only.

Final scientific interpretation of new outcomes remains to be reviewed before manuscript/reviewer-response integration. No DOI, public repository, package publication or portal submission is performed.

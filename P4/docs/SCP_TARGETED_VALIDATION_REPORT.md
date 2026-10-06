# P4 targeted software validation

Local author-performed validation, 2026-10-06. Software version 0.3.0.

## Workflow comparison

The protocol in SCP_TARGETED_VALIDATION_PROTOCOL.md was written before
execution. Each scenario supplies the same plan, contract, registry, worker
commands, input points and tolerance to all three workflows. The strict
script validates independently of P4's executor; it is not required to lose.

| Disclosed scenario | Exit-only control | Strict script | P4 |
| --- | --- | --- | --- |
| Normal execution | CONFORMANT | CONFORMANT | CONFORMANT |
| Injected numerical bias | CONFORMANT | NONCONFORMANT | NONCONFORMANT |
| Missing worker | CONFORMANT | ABSTAIN | ABSTAIN |
| Invalid JSON | CONFORMANT | ABSTAIN | ABSTAIN |
| Wrong observation ID | CONFORMANT | ABSTAIN | ABSTAIN |
| Zero reported calls | CONFORMANT | ABSTAIN | ABSTAIN |
| Explicit non-evaluable observation | CONFORMANT | ABSTAIN | ABSTAIN |
| Modified plan seed | CONFORMANT | REFUSED | REFUSED |
| Changed contract tolerance | CONFORMANT | REFUSED | REFUSED |

Raw reports and process outputs: results/software_validation_20261006/workflows.
Both integrity controls were refused before P4 produced an execution report.
The strict script preserves raw outputs, worker-file/output hashes and input
bindings; it deliberately demonstrates that equivalent safeguards are possible
in custom orchestration. Exit-only is a weak control, not a pytest benchmark.
The contribution supported here is a reusable standard interface for these
checks. These nine controls do not establish detection rates, efficiency,
external productivity or broad superiority.

## Current API transfer

A fresh local venv installed the 0.3.0 wheel, NumPy 2.3.3 and SciPy 1.16.3.
P4 was imported from site-packages. The new solve_ivp worker checked analytic
trajectories, amplitude scaling and half-life event localization. All three
passed; injecting a trajectory bias produced one FAIL and NONCONFORMANT.
Both runs executed all three selected relations and reported ten solver
calls. The example-specific change set is outside the compiler/executor.

Reports: results/software_validation_20261006/ode. This is developer-performed
transfer to a different numerical API, not third-party adoption or an eighth
historical defect family. The frozen seven-family set remains unchanged.
Final-archive extraction checks and public CI are recorded separately rather
than being assumed from these source-tree runs.

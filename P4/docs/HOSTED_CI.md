# Hosted CI

The repository workflow at `.github/workflows/p4-protocol-smoke.yml` provides
the hosted CPU check for the P4 protocol artifact. It is intentionally a
smoke-and-contract workflow rather than a rerun of every long experiment.

## What the workflow checks

1. The protocol modules compile under Python 3.11.
2. A non-MLSL differentiable propagator passes the generic interface smoke
   test.
3. The public registry and profile APIs load the checked-in evidence bundle.
4. The package installs from `P4/pyproject.toml`.
5. The P4 unit tests pass.
6. The generated smoke and registry JSON files are uploaded as a workflow
   artifact for inspection.

The runner uses CPU-only PyTorch. GPU timing remains a local hardware profile,
and the public-data experiments remain versioned evidence rather than a
per-commit CI job. This separation keeps CI deterministic and bounded while
preserving the distinction between software-contract checks and scientific
benchmark execution.

## Verified run

The workflow passed on 2026-08-12 for software commit `20fab75`:

- Repository: <https://github.com/hzhooning-art/DFSC>
- Workflow run: <https://github.com/hzhooning-art/DFSC/actions/runs/31581391771>

The tested commit identifies the protocol package and CI configuration used
for this hosted check. Later manuscript or experiment commits should be
recorded separately rather than silently attributed to this run.

## How to activate it

Push the repository to GitHub with the workflow at
`.github/workflows/p4-protocol-smoke.yml`. A run is triggered by changes under
`P4/` or by changes to the workflow itself. The hosted run should be cited as
an environment check; it does not replace the numerical evidence recorded in
`P4/results/`.

## Local preflight

From the repository root, run the same core checks with the project
environment:

```powershell
& .\P1\.venv\Scripts\python.exe .\P4\experiments\p4_generic_protocol_smoke.py
& .\P1\.venv\Scripts\python.exe .\P4\experiments\p4_public_api_smoke.py
& .\P1\.venv\Scripts\python.exe -m unittest discover -s .\P4\tests -p 'test_*.py' -v
```

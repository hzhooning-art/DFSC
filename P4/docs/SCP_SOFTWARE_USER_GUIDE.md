# P4 Testing: user guide for the SCP Software Track draft

This guide separates portable software validation from the seven historical
studies. The version 0.3.0 release tag is p4-scp-osp-v0.3.0; verify its public
publication status in the release receipt. A new Zenodo version DOI has not
been reserved. Do not reuse an older DOI as this snapshot's identifier.

## Install and try the portable path

Use Python 3.10 or later from the repository root:

```powershell
python -m pip install ./P4
python P4/scripts/verify_software_install.py
```

The second command creates temporary JSON reports and checks three outcomes:
a conformant `math.expm1` value/derivative run, a nonconformant run with an
injected worker bias, and rejection of a modified plan. It should print
`P4 installed-CLI smoke: positive, negative control, and tamper refusal passed`.
The example uses the Python standard library only. Version 0.3.0 moves PyTorch
to the optional protocol extra: use `python -m pip install ./P4[protocol]`
for the legacy dfsc_protocol API. Historical workers require their own pinned
scientific-library environments.

For the same-condition workflow comparison and the current SciPy ODE example,
see SCP_TARGETED_VALIDATION_REPORT.md and SCP_ODE_TRANSFER_GUIDE.md.

## Compile and execute a plan

From the repository root, after installing the package:

```powershell
p4-testing compile --contract P4/contracts/examples/expm1_execution_demo.json --risk-profile P4/contracts/examples/expm1_execution_risk.json --seed 17 --max-invocations 2 --wall-seconds 10 --output plan.json
p4-testing execute --contract P4/contracts/examples/expm1_execution_demo.json --plan plan.json --registry P4/contracts/examples/expm1_execution_registry.json --artifact-root P4 --output execution.json
```

`plan.json` records selected relation IDs, seeds, a digest, and declared risk
coverage. `execution.json` records per-relation observations, hashes, timing,
worker-reported calls, and a final status. `--max-invocations` means selected
**relation slots**, not actual subject calls. A plan with status `ABSTAIN` is
not a passing test. `CONFORMANT` requires all selected relations to have valid
passes. A valid observed failure yields `NONCONFORMANT`; missing or
non-evaluable evidence cannot yield `CONFORMANT`.
`--artifact-root` resolves `{p4_root}` in the trusted command registry and sets
the worker working directory. Specify it explicitly when running from an
installed wheel; otherwise a default rooted at the installed package would
not find repository-side example workers.

Never execute a registry received from an untrusted party. It contains local
commands that can launch programs. Review contracts, risk profiles, registry
paths, and worker code before use. The tool checks serialization and process
results, not the truth of a numerical oracle.

## Historical reproduction

The [coverage ledger](LIVE_RELATION_COVERAGE.md) links all seven admitted
cases to their per-case guides and saved reports. The frozen set is four SciPy,
one JAX, and two PyTorch defect families. These old and new upstream runtimes
must be provisioned separately; ignored local interpreters and unredistributable
wheels are **not** in the public package. Start with the portable example,
then follow a per-case guide only if its pinned environment can be recreated.
Saved reports are inspectable when old runtimes cannot be rebuilt, but they do
not replace an independently rerun experiment. Do not count repeated seeds,
relations, or injected biases as new historical defect families.

## Troubleshooting

- `p4-testing` not found: verify the package was installed into the active
  interpreter; use `python -m p4_testing` to bypass a missing console-script
  path.
- Plan digest mismatch: recompile from the same contract, risk profile, seed,
  and budget. Do not edit a plan by hand.
- Missing worker or timeout: inspect the report's process fields and trusted
  registry path; do not reinterpret a non-evaluable result as a pass.
- Historical package import failure: check the case-specific Python and
  package versions before running its worker.

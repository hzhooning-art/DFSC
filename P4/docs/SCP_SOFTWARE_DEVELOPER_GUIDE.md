# P4 Testing: developer and extension guide

## Components and boundaries

`src/p4_testing/contracts.py` validates versioned contracts and risk
profiles. `compiler.py` deterministically selects declared relations and
creates a coverage certificate. `cli.py` provides the `compile`, `execute`,
`replay`, and `audit` entry points. `execution.py` validates the plan against
its contract, launches registry commands without a shell, and collects typed
worker observations and audit hashes. `results.py` and the reconstruction
modules analyze previously captured benchmark evidence. The numerical
correctness of a new relation remains a worker-author obligation.

The command registry is trusted operator configuration, not untrusted data.
Its commands may execute arbitrary code. Supported substitutions and the
worker-observation schema are specified in [PLAN_EXECUTION_SLICE.md](PLAN_EXECUTION_SLICE.md).
An installed package does not contain repository-side example workers, so
invoke `execute` with `--artifact-root` pointing to the checked-out P4 tree
or another reviewed worker directory. The CLI verifies that directory exists.
The executor's file and command hashes bind records for audit; they are not
signatures or protection against a malicious local operator.

## Add a new component relation

1. Define the component domain and candidate relation in a JSON contract
   conforming to `contracts/p4-component-contract-v1.schema.json`.
2. Declare only mechanisms the relation actually checks. Write a separate
   risk profile with mandatory mechanisms; an unsupported mechanism should
   cause abstention, not be silently mapped to another check.
3. Implement a worker emitting one `P4-Relation-Observation-v1` JSON object
   with the exact selected `test_id`, `PASS`, `FAIL`, or `NON_EVALUABLE`, a
   nonnegative integer `subject_invocations`, and a `details` object.
4. Bind the worker through a local, reviewed registry and compile/execute a
   plan using the commands in the user guide. Preserve the raw report.
5. Add tests for a valid pass, a valid failure, malformed or missing output,
   timeout, and edited-plan refusal. Review any external numerical reference
   independently of the executor.

## Tests and release checks

The `ode_decay` example adds three SciPy relations without modifying the
compiler/executor or its mechanism vocabulary. Workers read contract
tolerances; commands and input files are bound by report hashes. Run
`scripts/verify_ode_transfer.py` with the optional `ode-example` extra.
The nine predeclared workflow controls compare P4 with both an exit-only
control and a strict independently written script. The strict script matches
all nine P4 decisions; this is interface/transfer evidence, not superiority.
See `SCP_TARGETED_VALIDATION_PROTOCOL.md` and `SCP_TARGETED_VALIDATION_REPORT.md`.

Run the installed CLI smoke (`python P4/scripts/verify_software_install.py`)
and executor unit tests (`python -m unittest discover -s P4/tests/reconstruction -p 'test_*execution*unittest.py'`).
The dedicated CI workflow builds a wheel, installs it, then runs the smoke
and executor tests. A public OSP release additionally needs an immutable tag,
the verified CI result for that tag, a source archive, a license, documentation,
and a newly deposited artifact DOI. Development-tree success does not prove
that a reviewer can rebuild the historical upstream environments.

The seven historical case guides and
`results/reconstruction/live_relation_coverage.json` are the evidence index.
The benchmark's unit of analysis is the defect family; repeated runs and
controlled mutations must not be appended as independent historical defects.

The review-artifact worker controls use mocked role outputs and the active
interpreter only for input-file validation; these do not run historical
scientific libraries. Runtime-dependent integration tests remain separate and
skip when their environments are unavailable. Run the tests from an extracted
artifact as well as the development tree; a local-only path must not make a
mocked-output unit test require private historical interpreters. See
[historical reproduction boundary](SCP_HISTORICAL_REPRODUCTION_BOUNDARY.md)
for environment requirements and legacy frozen-metadata clarifications.

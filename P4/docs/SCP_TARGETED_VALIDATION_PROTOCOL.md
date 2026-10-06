# Targeted software validation protocol

Defined before running the new workflow comparison, 2026-10-06.
This is a local engineering protocol, not a registered or blinded study.

## Fixed scope

Use the same two math.expm1 workers, points, thresholds, selected relation
IDs, seed and 10-second wall budget for three workflows:

1. Exit-only: run each available command, retain raw outputs and return codes,
   and admit if all launched processes exit zero. This is explicitly a weak
   process-success control, not a characterization of pytest.
2. Strict scripted: independently validate the JSON plan digest, contract
   declarations, requested coverage, required observation IDs, schema,
   finite details, positive reported calls, complete execution and outcomes.
   Retain raw outputs. This strong control is permitted to match P4.
3. P4: use the installed compile/execute CLI and its standard evidence report.

Scenarios: normal execution, injected numerical bias, missing worker,
invalid JSON, wrong observation ID, zero reported calls, explicit
NON_EVALUABLE, modified plan seed, and changed contract tolerance.
The last two are inconsistent input records, not numerical defects.
Expected decisions are CONFORMANT, NONCONFORMANT, five ABSTAIN outcomes,
then two REFUSED outcomes, respectively. Compare decisions and retained
fields. Do not measure or claim detection superiority, runtime gains,
developer productivity, security against an adversarial local operator,
or field-wide rates. Do not tune scenarios after observing results without
recording a protocol change.

## Transfer example

Integrate SciPy 1.16.3 solve_ivp on the scalar decay equation y'=-k*y.
Check analytic trajectories, amplitude scaling and terminal half-life events
through three separately selected relations. Test a value bias as a negative
control. Install the built P4 wheel into a clean environment, run from an
extracted artifact directory with PYTHONPATH empty, and record interpreter,
package versions, imported module location and input/worker hashes.
The contract, configuration, registry and worker are example-side additions;
the compiler and executor must not require example-specific modifications.
This is author-performed API transfer, not independent external adoption,
a new historical defect family or a test of stiff/nonlinear ODE solvers.

## Admission rule

Publish results only if both positive examples pass, both negative controls
produce numerical failure, all integrity refusals prevent dispatch, and the
strict comparator agrees with expected decisions. Publish disagreements and
repair implementation errors before rebuilding the final snapshot. Missing
historical runtimes remain explicit skips, not reproduction successes.

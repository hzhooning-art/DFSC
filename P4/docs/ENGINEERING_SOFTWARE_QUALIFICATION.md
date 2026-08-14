# Engineering Software Qualification Workflow

This document connects the P4 numerical evidence to a repeatable software
acceptance process. A component is qualified only for its declared domain,
reference, dtype, device, and tolerance set.

## Relationship to model and simulation V&V

This workflow is a component-level check, not an end-to-end validation of a
physical model or simulation. It verifies the executed numerical map,
parameter-derivative path, interface behavior, and selected compositions before
or during integration. A passing component record does not replace model-form
assessment, uncertainty quantification, validation against experiments, or an
intended-use decision. It supplies auditable evidence for one software layer
inside those broader engineering V&V activities.

## Required record

1. Declare input and parameter domains, output shape, dtype, and device policy.
2. Freeze independent value and directional-derivative references.
3. Freeze value and gradient tolerances before executing the candidate.
4. Run finite-value, gradient, batch-shape, and device-locality checks.
5. Apply `qualify_audit` and retain every failed gate and warning.
6. Add calibration, composition, OOD, horizon, and performance evidence to the
   registry when those claims are needed by the downstream application.

## CI decision

`conformant` permits integration only with the declared scope attached to the
component record. A nonempty scope field can be displayed as "conformant with
declared scope limits," but it remains the same machine state.
`nonconformant` blocks release while preserving the JSON diagnostics. An empty
gate set is `incomplete` and cannot be presented as a qualification result.

## Minimal run

```powershell
python P4/examples/qualify_exponential_component.py
python -m unittest discover -s P4/tests -p "test_*.py" -v
```

The example is deliberately small so that interface and refusal semantics can
be checked without the manuscript experiments. The paper-scale records under
`P4/results` provide the multi-backend and hardware evidence.

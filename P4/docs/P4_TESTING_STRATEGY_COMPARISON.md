# P4 Paired Testing-Strategy Comparison

## Question

What fault-class coverage is added when component qualification moves from
value/gradient assertions to numerical-property checks, execution evidence,
and a complete machine-readable evidence record?

## Frozen design

- Ten fault classes from the P4 conformance catalogue.
- Twenty implementation trials per class, for 200 injected records.
- Forty unmodified records for false-rejection checks.
- Every strategy receives the same records and mutations.
- Fault class is the primary comparison unit.

## Results

| Strategy | Declared checks | Classes detected | Instances detected | Clean rejected |
|---|---:|---:|---:|---:|
| Envelope plus value/gradient | 3 | 3/10 | 60/200 | 0/40 |
| Numerical-property suite | 8 | 5/10 | 100/200 | 0/40 |
| Execution-aware suite | 13 | 8/10 | 160/200 | 0/40 |
| Executable evidence record | 17 | 10/10 | 200/200 | 0/40 |

The progression isolates the contribution of evidence beyond numerical value
and gradient checks. Batch/repeatability/scope properties, execution metadata,
and frozen-scope/profile rules each close fault classes that earlier strategies
cannot observe. The initial envelope already requires a provenance field;
complete records impose the remaining record semantics.

## Claim boundary

This is a paired comparison on a declared catalogue. It does not estimate
field-defect prevalence, prove complete fault detection, or represent every
possible unit, property-based, or metamorphic testing system. The 20 repeated
mutations within a class are implementation trials, not 20 independent fault
families.

## Reproduction

```bash
python P4/experiments/p4_testing_strategy_comparison.py
python -m unittest P4.tests.test_testing_strategy_comparison -v
```

The machine-readable result is
`P4/results/p4_testing_strategy_comparison.json`.

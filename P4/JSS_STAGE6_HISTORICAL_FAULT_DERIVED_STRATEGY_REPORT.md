# JSS Stage 6: Historical-Fault-Derived Strategy Discrimination

## Design

SciPy issue #8906 reported that the 1x1 special case in `solve_banded`
selected band row one rather than the row indexed by the declared upper
bandwidth. A compact source-derived surrogate preserves that reported faulty
choice. The comparison pairs two mathematically equivalent representations of
the same 1x1 system: a padded `(1, 1)` representation that masks the fault and
the conventional `(0, 0)` representation that exposes it. Thirty-two
right-hand-side variants are replayed through the same pair.

## Result

| Strategy or control | Detected/passed |
|---|---:|
| Weak padded-example test: fault detections | 0/32 |
| Representation-equivalence property: fault detections | 32/32 |
| Current SciPy 1.18.0 fixed control: property passes | 32/32 |

The result supplies a concrete historical motivation for representation-
equivalence testing: a numerically correct example can exercise the faulty
implementation without exposing the defect, while an equivalent encoding
changes the observable outcome.

## Claim boundary

The 32 inputs are paired variants of one upstream defect family, not 32
independent defects. More importantly, the faulty function is a compact
source-derived surrogate rather than an installed historical SciPy release.
The experiment therefore establishes strategy discrimination on a real-
defect-derived mutation, but the complete buggy/fixed environment-pair count
remains zero. Full records are in
`results/p4_historical_fault_derived_strategy.json`.

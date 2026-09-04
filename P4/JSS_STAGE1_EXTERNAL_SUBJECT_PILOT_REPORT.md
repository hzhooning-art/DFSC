# P4 JSS Stage 1: External-Subject Pilot

Updated: 2026-09-03

## Objective

Establish that the P4 testing layers can be executed against independently
maintained numerical software rather than only against DFSC-owned backends.
This stage is an adapter and design pilot for the planned cross-project study;
it is not itself sufficient evidence for a Journal of Systems and Software
submission.

## Frozen design

- Systems under test: three differentiable PyTorch component interfaces:
  `torch.matrix_exp`, `torch.linalg.solve`, and `torch.lgamma`.
- Independent value-reference route: SciPy/NumPy.
- Eight behavior-level fault families.
- Ten deterministic trials per subject-fault combination.
- 24 subject-fault combinations, 240 injected trials, and 30 clean controls.
- Primary experimental unit: subject-fault combination, not injected trial.
- Common records are evaluated by five cumulative testing strategies.

## Results

| Strategy | Detected trials | Fully detected subject-fault combinations | Clean rejected |
|---|---:|---:|---:|
| Single-point unit test | 60/240 (25.00%) | 6/24 | 0/30 |
| Full-batch value test | 90/240 (37.50%) | 9/24 | 0/30 |
| Value and gradient test | 177/240 (73.75%) | 16/24 | 0/30 |
| Numerical-property suite | 238/240 (99.17%) | 23/24 | 0/30 |
| Execution-evidence suite | 240/240 (100.00%) | 24/24 | 0/30 |

The silent float32 downgrade was often already visible through directional
gradient disagreement.  The dtype check remains necessary because two of its
30 subject-seed trials stayed within the numerical-property tolerances.  This
overlap is retained rather than forcing each layer to own an artificial,
disjoint fault class.

## Claim boundary

PyTorch is one vendor codebase even though three APIs are exercised.  SciPy is
an independent reference implementation, not a second system under test.  The
faults are controlled behavior mutations rather than historical field defects.
Repeated seeds within one subject-fault combination are not independent fault
families.  The result establishes executable external adapters and a usable
experimental design; it does not establish cross-project generality.

## JSS continuation gate

The next stage must add at least two more independently maintained systems
under test, generic or source-level mutation baselines, and historical defects
or fixed-version regressions.  Results must be reported by project and by fault
family, including leave-one-project-out heterogeneity and runtime cost.  If
those data cannot be obtained, P4 should return to the Software Quality Journal
route rather than treating this pilot as JSS-level evidence.

## Reproduction

```console
python P4/experiments/p4_external_subject_pilot.py
python -m unittest P4.tests.test_external_subject_pilot -v
```

The aggregate record is `P4/results/p4_external_subject_pilot.json`.

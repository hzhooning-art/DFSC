# JSS Stage 7: Cross-Project External-Subject Benchmark

## Design

The same seven output and execution fault families are applied to one
production numerical interface from each of three independently maintained
projects: `numpy.linalg.solve`, `scipy.linalg.expm`, and `torch.lgamma`.
Twelve seeds per project--fault pair give 252 injected trials and 36 clean
controls. The primary comparison unit is the 21 project--subject--fault
clusters.

## Result

| Strategy | Detected trials | Fully detected clusters | Clean rejects |
|---|---:|---:|---:|
| Single-point unit test | 72/252 (28.57%) | 6/21 | 0/36 |
| Full-batch value test | 144/252 (57.14%) | 12/21 | 0/36 |
| Numerical-property suite | 216/252 (85.71%) | 18/21 | 0/36 |
| Execution-evidence suite | 252/252 (100%) | 21/21 | 0/36 |

The controlled independent-SUT count is now three, and the monotone strategy
ordering is not confined to repeated interfaces from one vendor.

## Claim boundary

The injected wrappers are controlled faults, not historical field defects.
Repeated seeds do not increase the project or fault-family count. The NumPy
and SciPy subjects do not evaluate automatic differentiation. This stage
closes the three-project controlled-coverage gate; it does not close the
separate historical buggy/fixed-pair gate. Complete records are stored in
`results/p4_cross_project_external_subjects.json`.

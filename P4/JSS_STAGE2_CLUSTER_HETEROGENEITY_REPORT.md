# P4 JSS Stage 2: Cluster and Interface Heterogeneity Audit

Updated: 2026-09-03

## Objective

Test whether the Stage 1 aggregate mutation result is stable across numerical
interfaces and fault families. The primary unit is the 24 subject-fault
combinations; ten repeated seeds within a combination are not treated as ten
independent defects. This is a reanalysis of the Stage 1 pilot and adds no new
project or historical defect.

## Analysis

- Detection is stratified by interface and fault family.
- Uncertainty is obtained by 10,000 deterministic bootstrap resamples of the 24
  subject-fault clusters.
- Adjacent cumulative strategies are compared on fully detected clusters.
- Each interface is omitted in turn to check whether the aggregate ordering is
  driven by one API.
- Clean-control false rejection retains its finite-sample Wilson interval.

## Cluster-level result

| Strategy | Mean cluster detection | 95% cluster-bootstrap interval | Fully detected clusters | Undetected clusters |
|---|---:|---:|---:|---:|
| Single-point unit test | 0.2500 | [0.0833, 0.4167] | 6/24 | 18/24 |
| Full-batch value test | 0.3750 | [0.2083, 0.5833] | 9/24 | 15/24 |
| Value and gradient test | 0.7375 | [0.5625, 0.9042] | 16/24 | 6/24 |
| Numerical-property suite | 0.9917 | [0.9750, 1.0000] | 23/24 | 0/24 |
| Execution-evidence suite | 1.0000 | [1.0000, 1.0000] | 24/24 | 0/24 |

The successive layers newly fully detect 3, 7, 7, and 1 subject-fault
clusters, respectively, with no cluster regression at any step. The last gain
is the `torch_lgamma_shift` silent-dtype-downgrade cluster, which confirms that
explicit execution evidence contributes information not guaranteed by the
numerical-property layer.

## Interface heterogeneity

The complete suite detects 80/80 injected trials for each of matrix
exponential action, linear solve, and shifted log-gamma. The numerical-property
suite detects 80/80, 80/80, and 78/80, respectively. When each interface is
omitted in turn, complete-suite detection remains 1.0000 and the preceding
strategy remains between 0.9875 and 1.0000. Thus the Stage 1 ordering is not
caused by one unusually easy interface within this vendor.

All 30 clean controls pass the complete suite. The observed false-rejection
rate is zero, but its Wilson 95% interval is [0, 0.1135]; the finite sample does
not justify claiming a universal zero false-rejection probability.

## JSS readiness decision

The statistical aggregation is now appropriate for the pilot, and the layered
gain is stable across its three interfaces. Nevertheless the JSS readiness
gate remains failed: observed independent SUT projects = 1 versus a declared
minimum of 3, and historical defect families = 0. Repeated APIs, seeds, and
bootstrap samples do not increase either count.

The next empirical stage must ingest fixed-version regressions or independently
maintained source mutants from at least two additional projects. Until then,
this result strengthens internal validity only, not cross-project external
validity.

## Reproduction

```console
python P4/experiments/p4_external_subject_heterogeneity.py --stdout-summary
python -m unittest P4.tests.test_external_subject_heterogeneity -v
```

The aggregate artifact is
`P4/results/p4_external_subject_heterogeneity.json`.

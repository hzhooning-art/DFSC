# P4: Executable Conformance Specification for Differentiable Components

P4 provides a backend-independent engineering-software workflow for qualifying
differentiable numerical components on a declared domain. It separates numerical
value fidelity, parameter-gradient credibility, calibration, module reuse,
OOD and long-horizon behavior, and hardware profiling into auditable gates.
Predeclared error tolerances produce deterministic conformance outcomes and
machine-readable failure reasons suitable for continuous integration.

The project-local `DFSC-DNC-Conformance-v2` specification defines Core,
Extended, and Application profiles. A standalone JSON Schema, deterministic
canonicalization, SHA-256 record digests, legacy-record migration, and matching
Python/CLI adapters make conformance records portable across tool boundaries.
This specification draws on established software-quality and testing concepts;
it is not represented as an ISO, IEC, or IEEE standard.

The current evidence covers four backend families: a Mittag--Leffler spectral
layer, matrix-exponential actions, fixed-step linear RK4, and nonlinear
Logistic RK4. A direct 2D periodic heat-equation audit and public-data transfer
records extend the protocol beyond scalar smoke tests. Backend-specific metrics
must not be interpreted as a universal solver ranking.

## Layout

- `dfsc_protocol/`: public audit and registry API.
- `spec/`: versioned JSON Schema, profile requirements, and example records.
- `experiments/`: scripts used by the current manuscript evidence.
- `results/`: machine-readable JSON audit records.
- `tests/`: CPU-compatible contract and 2D heat-equation checks.
- `examples/`: minimal executable component-qualification examples.
- `docs/ENGINEERING_SOFTWARE_QUALIFICATION.md`: CI and integration workflow.
- `paper/`: synchronized manuscript sources, bibliography, figures, and data
  assembly scripts.

## Minimal verification

```bash
python -m pip install torch
python -m unittest discover -s P4/tests -p "test_*.py" -v
python P4/experiments/p4_generic_protocol_smoke.py
python P4/experiments/p4_public_api_smoke.py
python P4/examples/qualify_exponential_component.py
python -m dfsc_protocol.cli record.json --output evaluated.json
python P4/experiments/p4_csi_conformance_validation.py
python P4/experiments/p4_testing_strategy_comparison.py
```

The precision, calibration, plotting, and selected baseline scripts additionally
use NumPy, SciPy, and Matplotlib. GPU profile records state their device and
dtype and are not reproduced by the hosted CPU workflow.

## Manuscript data

```bash
python P4/paper/build_paper_data.py
python P4/paper/make_figures.py
```

`paper/paper_data.json` records the source JSON files used by manuscript tables
and figures. Public-data summary records include dataset DOI, license, split,
and source-result provenance. Regenerating the upstream public-data fits
requires their dataset files and the corresponding DFSC propagation code;
the normalization script can also operate directly on the included result
records without a machine-specific path.

## Paired testing-strategy comparison

The frozen ten-class mutation catalogue is also replayed through four
progressively stronger strategies. Fault-class coverage rises from 3/10 for
envelope plus value/gradient checks to 5/10 for a numerical-property suite,
8/10 for an execution-aware suite, and 10/10 for the complete executable
evidence record. All four strategies accept the 40 clean controls. The design,
result table, and claim boundary are documented in
`docs/P4_TESTING_STRATEGY_COMPARISON.md`; machine-readable results are in
`results/p4_testing_strategy_comparison.json`.

## JSS external-subject pilot

The first JSS-oriented pilot executes the layered comparison against three
PyTorch numerical-component interfaces with SciPy/NumPy value references.  On
240 behavior-fault trials, detection rises from 25.00% for a single-point unit
test to 73.75% after directional-gradient evidence, 99.17% for the numerical-
property suite, and 100% for the execution-evidence suite; none of 30 clean
controls is rejected.  This remains a single-vendor pilot, not a cross-project
or historical-defect result.  See `JSS_STAGE1_EXTERNAL_SUBJECT_PILOT_REPORT.md`.

## JSS cluster-heterogeneity audit

The Stage 1 trials are now reanalysed with the 24 subject-fault combinations as
clusters. Complete execution evidence detects all 24 clusters; successive
testing layers add 3, 7, 7, and 1 fully detected clusters without a regression.
Leave-one-interface-out detection remains 1.000 for the complete suite, so the
aggregate result is not driven by one of the three PyTorch APIs. This improves
internal validity but leaves the JSS external-validity gate failed because the
study still has one independent SUT project and no historical defects. See
`JSS_STAGE2_CLUSTER_HETEROGENEITY_REPORT.md`.

## JSS historical-defect intake

Stage 3 establishes a strict upstream-defect admission rule and inventories
four candidates from PyTorch, JAX, and PyBNF. The installed PyTorch version
passes fixed-behavior replays for two closed wrong-gradient reports, including
one CUDA/non-contiguous case. Neither reported buggy PyTorch version is
installed, and the other project dependencies are unavailable, so the number
of complete buggy/fixed pairs remains zero. Candidate provenance is not counted
as empirical defect detection. See
`JSS_STAGE3_HISTORICAL_DEFECT_INTAKE_REPORT.md`.

## JSS cross-version replay harness

Stage 4 freezes one dependency-minimal runner for the PyTorch #80770 and #30303
buggy/fixed comparisons. Under the installed PyTorch 2.11.0+cu128, both current
fixed roles are numerically confirmed and captured with the runner SHA-256.
No legacy Python, container runtime, or cached old package is available, so the
buggy roles and complete-pair count remain zero. See
`JSS_STAGE4_CROSS_VERSION_REPLAY_HARNESS_REPORT.md` and
`docs/P4_HISTORICAL_REPLAY_MATRIX.md`.

## JSS cross-project fixed-side regression replay

Stage 5 adds the installed SciPy regression case for upstream issue #8906 to
the two frozen PyTorch replays. The current SciPy 1.18.0 implementation solves
both 1x1 band representations exactly and preserves the right-hand side, so
three fixed-side cases are now confirmed across two independent projects. No
reported buggy SciPy release was executed: buggy roles and complete pairs both
remain zero, and this is not counted as historical-defect detection. See
`JSS_STAGE5_CROSS_PROJECT_FIXED_REGRESSION_REPORT.md`.

## JSS historical-fault-derived strategy discrimination

Stage 6 converts the reported row-selection error in SciPy #8906 into a compact
source-derived faulty implementation and compares a weak padded example with a
representation-equivalence property on 32 paired inputs. The weak example
detects 0/32 faults; the equivalence property detects 32/32, while current
SciPy 1.18.0 passes all 32 fixed controls. This connects the proposed testing
logic to one real upstream defect family, but it does not execute an old SciPy
release and does not increase the complete buggy/fixed-pair count. See
`JSS_STAGE6_HISTORICAL_FAULT_DERIVED_STRATEGY_REPORT.md`.

## JSS cross-project external-subject benchmark

Stage 7 applies one common seven-fault catalogue to production interfaces from
NumPy, SciPy, and PyTorch. Across 252 injected trials, detection rises from
28.57% for a single-point test to 57.14% for full-batch values, 85.71% for
numerical properties, and 100% for execution evidence; all 36 clean controls
pass. All 21 project--subject--fault clusters are detected by the complete
suite. This closes the controlled three-project coverage gate, but injected
wrappers are not historical field defects. See
`JSS_STAGE7_CROSS_PROJECT_EXTERNAL_SUBJECT_REPORT.md`.

## JSS complete historical pair

Stage 8 executes the unchanged frozen runner against the official PyTorch
1.11.0+cpu wheel and current PyTorch 2.11.0+cu128. The old version reproduces
the reported zero derivative for `xlogy(0, 2)`; the current version returns
`log(2)` exactly at stored precision. Both environment roles are confirmed,
raising the complete historical buggy/fixed-pair count from zero to one. See
`JSS_STAGE8_COMPLETE_HISTORICAL_PAIR_REPORT.md`.

## JSS second complete historical pair

Stage 9 executes another unchanged runner in the official SciPy 1.14.1 wheel
and current SciPy 1.18.0. The historical version raises the reported 1x1
`solve_banded` indexing error for the conventional band representation while
an equivalent padded representation masks it; the current version solves both
exactly. Both roles are confirmed with runner and wheel hashes, raising the
complete-pair count to two across two projects. See
`JSS_STAGE9_SECOND_COMPLETE_HISTORICAL_PAIR_REPORT.md`.

## JSS third complete historical defect family

Stage 11 executes a third frozen runner against the same official SciPy
1.14.1 wheel and current SciPy 1.18.0. For upstream issue #15620, the old
release silently returns all-zero `resample_poly` output for both `int16` and
`int32` inputs although the float64 reference reaches 3.001552; the current
release matches that reference with zero maximum error. This raises the total
to three complete historical families across two projects and diversifies the
observed failures to wrong derivative, exception, and silent wrong output. See
`JSS_STAGE11_THIRD_HISTORICAL_DEFECT_FAMILY_REPORT.md`.

## Scope

Passing the protocol qualifies one implementation on its declared domain. It
does not establish universal solver accuracy, downstream predictive benefit,
or reliability on untested geometries, discretizations, dtypes, and devices.

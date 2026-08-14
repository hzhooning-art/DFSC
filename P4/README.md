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

## Scope

Passing the protocol qualifies one implementation on its declared domain. It
does not establish universal solver accuracy, downstream predictive benefit,
or reliability on untested geometries, discretizations, dtypes, and devices.

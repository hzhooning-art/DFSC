# DFSC-DNC-Conformance-v3

This directory publishes the project-local interchange specification used by
the P4 implementation. It is a proposed conformance format, not an ISO, IEC,
or IEEE standard.

Profiles are cumulative:

- `core`: value and gradient accuracy, batch shape and independence, and repeatability.
- `extended`: Core plus OOD and long-horizon checks, dtype/device conformance,
  unit consistency, and resource reporting.
- `application`: Extended plus calibration and composition checks.

An implementation conforms only when every check required by the selected
profile is present and true. Requested and observed dtype/device values are
compared by the evaluator. Canonical JSON uses sorted keys and compact
separators; its SHA-256 digest identifies the evaluated record.

The evaluated operating scope must be frozen before candidate execution. Each
profile also requires a minimum number of cases and named coverage anchors:

- `core`: at least 8 cases covering nominal, boundary, and heterogeneous-batch behavior.
- `extended`: at least 16 cases, adding perturbation, execution-policy, and long-horizon anchors.
- `application`: at least 24 cases, adding an application-composition anchor.

These requirements prevent a record from obtaining conformance by declaring an
arbitrarily narrow scope. Migration from v1 or v2 preserves scientific evidence
for audit, but the migrated record is nonconformant until its scope is frozen and
the v3 coverage requirements are rerun.

Validate a record with:

```bash
python -m dfsc_protocol.cli record.json --output evaluated.json
```

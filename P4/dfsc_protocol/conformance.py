"""Executable conformance profiles for differentiable numerical components.

The module defines a proposed, project-local specification. It maps existing
software-quality and testing concepts to evidence produced by differentiable
numerical components; it is not an ISO, IEC, or IEEE standard.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


CONFORMANCE_SCHEMA = "DFSC-DNC-Conformance-v3"
LEGACY_SCHEMA = "DFSC-DNC-Conformance-v2"
EARLIEST_LEGACY_SCHEMA = "DFSC-DNC-Conformance-v1"

PROFILE_REQUIREMENTS = {
    "core": (
        "value_accuracy",
        "gradient_accuracy",
        "batch_shape",
        "batch_independence",
        "repeatability",
    ),
    "extended": (
        "value_accuracy",
        "gradient_accuracy",
        "batch_shape",
        "batch_independence",
        "repeatability",
        "ood_control",
        "long_horizon",
        "dtype_conformance",
        "device_local",
        "unit_consistency",
        "resource_reported",
    ),
    "application": (
        "value_accuracy",
        "gradient_accuracy",
        "batch_shape",
        "batch_independence",
        "repeatability",
        "ood_control",
        "long_horizon",
        "dtype_conformance",
        "device_local",
        "unit_consistency",
        "resource_reported",
        "calibration",
        "composition",
    ),
}

PROFILE_COVERAGE_REQUIREMENTS = {
    "core": {
        "minimum_samples": 8,
        "required_anchors": ("nominal", "boundary", "heterogeneous_batch"),
    },
    "extended": {
        "minimum_samples": 16,
        "required_anchors": (
            "nominal",
            "boundary",
            "heterogeneous_batch",
            "perturbation",
            "execution_policy",
            "long_horizon",
        ),
    },
    "application": {
        "minimum_samples": 24,
        "required_anchors": (
            "nominal",
            "boundary",
            "heterogeneous_batch",
            "perturbation",
            "execution_policy",
            "long_horizon",
            "application_composition",
        ),
    },
}

REQUIRED_TOP_LEVEL = {
    "schema",
    "component",
    "profile",
    "operating_domain",
    "coverage",
    "requested_execution",
    "observed_execution",
    "evidence",
    "provenance",
}


def migrate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a v1/v2 record while requiring v3 coverage requalification."""

    migrated = deepcopy(dict(record))
    if migrated.get("schema") == CONFORMANCE_SCHEMA:
        return migrated
    source_schema = migrated.get("schema")
    if source_schema not in {LEGACY_SCHEMA, EARLIEST_LEGACY_SCHEMA}:
        raise ValueError(f"unsupported conformance schema: {migrated.get('schema')!r}")
    migrated["schema"] = CONFORMANCE_SCHEMA
    migrated.setdefault("profile", "core")
    migrated.setdefault("requested_execution", {"dtype": "unspecified", "device": "unspecified"})
    migrated.setdefault("observed_execution", deepcopy(migrated["requested_execution"]))
    migrated.setdefault("provenance", {"implementation": "legacy", "run_id": "legacy-import"})
    migrated["provenance"].setdefault("migrated_from", source_schema)
    migrated["coverage"] = {
        "scope_frozen": False,
        "sample_count": 0,
        "anchors": [],
        "migration_requires_requalification": True,
    }
    return migrated


def _coverage_failures(record: Mapping[str, Any], profile: str) -> list[str]:
    coverage = record["coverage"]
    rules = PROFILE_COVERAGE_REQUIREMENTS[profile]
    failures: list[str] = []
    if coverage.get("scope_frozen") is not True:
        failures.append("scope_not_frozen")
    sample_count = coverage.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise ValueError("coverage.sample_count must be an integer")
    if sample_count < rules["minimum_samples"]:
        failures.append("insufficient_samples")
    anchors = coverage.get("anchors")
    if not isinstance(anchors, list) or not all(isinstance(item, str) for item in anchors):
        raise ValueError("coverage.anchors must be a list of strings")
    missing_anchors = sorted(set(rules["required_anchors"]).difference(anchors))
    failures.extend(f"missing_anchor:{name}" for name in missing_anchors)
    if coverage.get("migration_requires_requalification") is True:
        failures.append("migration_requires_requalification")
    return failures


def _execution_checks(record: dict[str, Any]) -> None:
    requested = record["requested_execution"]
    observed = record["observed_execution"]
    evidence = record["evidence"]
    if requested.get("dtype") != "unspecified":
        evidence["dtype_conformance"] = requested.get("dtype") == observed.get("dtype")
    if requested.get("device") != "unspecified":
        evidence["device_local"] = requested.get("device") == observed.get("device")


def evaluate_conformance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a record and assign a deterministic conformance outcome."""

    result = migrate_record(record)
    missing = sorted(REQUIRED_TOP_LEVEL.difference(result))
    if missing:
        raise ValueError(f"record is missing top-level fields: {missing}")
    profile = result["profile"]
    if profile not in PROFILE_REQUIREMENTS:
        raise ValueError(f"unknown conformance profile: {profile!r}")
    if not result["component"].get("name") or not result["component"].get("version"):
        raise ValueError("component name and version are required")
    if not result["operating_domain"].get("identifier"):
        raise ValueError("operating_domain.identifier is required")
    if not result["provenance"].get("implementation") or not result["provenance"].get("run_id"):
        raise ValueError("provenance implementation and run_id are required")

    _execution_checks(result)
    required = PROFILE_REQUIREMENTS[profile]
    coverage_failures = _coverage_failures(result, profile)
    missing_evidence = [name for name in required if name not in result["evidence"]]
    failed = [name for name in required if result["evidence"].get(name) is False]
    invalid = [name for name in required if name in result["evidence"] and not isinstance(result["evidence"][name], bool)]
    if invalid:
        raise ValueError(f"conformance evidence must be boolean: {invalid}")

    result["conformance"] = {
        "profile": profile,
        "required_checks": list(required),
        "missing_checks": missing_evidence,
        "failed_checks": failed,
        "coverage_failures": coverage_failures,
        "status": (
            "conformant"
            if not missing_evidence and not failed and not coverage_failures
            else "nonconformant"
        ),
    }
    return result


def canonical_json(record: Mapping[str, Any]) -> str:
    """Return the normalized representation used across API and CLI adapters."""

    evaluated = evaluate_conformance(record)
    return json.dumps(evaluated, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def record_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()

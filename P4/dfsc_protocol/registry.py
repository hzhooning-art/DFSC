"""Stable, dependency-free access to primitive protocol artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "DFSC-Primitive-Protocol-v1"
PROFILE_SCHEMA = "DFSC-Primitive-Profile-v1"
REQUIRED_DIMENSIONS = ("value", "gradient", "calibration", "module_reuse", "ood", "long_horizon")
VALID_STATES = {"pass", "fail", "missing", "warning", "reported_prior_validation"}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate a primitive registry JSON artifact."""

    record = _read_json(path)
    validate_registry(record)
    return record


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a standardized runtime profile JSON artifact."""

    record = _read_json(path)
    validate_profile(record)
    return record


def validate_registry(record: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a registry violates the public schema."""

    if record.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"unexpected registry schema: {record.get('schema')!r}")
    if tuple(record.get("required_dimensions", ())) != REQUIRED_DIMENSIONS:
        raise ValueError("registry required_dimensions do not match protocol v1")
    records = record.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("registry records must be a non-empty list")
    for item in records:
        if not item.get("backend") or not item.get("family"):
            raise ValueError("each registry record requires backend and family")
        for dimension in REQUIRED_DIMENSIONS:
            if item.get(dimension) not in VALID_STATES and item.get(dimension) != "reported_prior_validation":
                raise ValueError(f"invalid state for {item['backend']}.{dimension}: {item.get(dimension)!r}")


def validate_profile(record: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a runtime profile violates profile v1."""

    if record.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"unexpected profile schema: {record.get('schema')!r}")
    rows = record.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("profile rows must be a non-empty list")
    required = {"backend", "batch", "mean_latency_ms", "samples_per_second", "outputs_finite"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"profile row is missing fields: {sorted(missing)}")
        if row["batch"] <= 0 or row["mean_latency_ms"] < 0 or row["samples_per_second"] < 0:
            raise ValueError("profile timing values must be non-negative")


def summarize_registry(record: dict[str, Any]) -> dict[str, Any]:
    """Return a compact backend/status summary for reports and notebooks."""

    validate_registry(record)
    return {
        "schema": record["schema"],
        "backends": [
            {
                "backend": item["backend"],
                "family": item["family"],
                "status": item["status"],
                "coverage": item["coverage"],
            }
            for item in record["records"]
        ],
    }

"""Schema helpers for independently logged strategy evaluations."""

from __future__ import annotations

from typing import Any


def independent_record(
    *,
    strategy: str,
    family_id: str,
    project: str,
    evidence_status: str,
    detected: bool,
    plan_digest: str,
    runner_sha256: str,
    coverage_certificate: dict[str, Any] | None = None,
    execution_log_status: str = "UNOBSERVED",
    execution_log_digest: str = "",
) -> dict[str, Any]:
    metadata_only = evidence_status == "captured_case_metadata"
    return {
        "strategy": strategy,
        "family_id": family_id,
        "project": project,
        "status": "ABSTAIN" if metadata_only else ("detected" if detected else "missed"),
        "reason_code": "NO_INDEPENDENT_EXECUTION_LOG" if metadata_only else None,
        "detected": False if metadata_only else bool(detected),
        "evidence_status": evidence_status,
        "wall_time_seconds": None,
        "subject_invocations": None,
        "time_to_first_detection_seconds": None,
        "plan_digest": plan_digest,
        "runner_sha256": runner_sha256,
        "coverage_certificate": coverage_certificate or {
            "requested": [],
            "covered": [],
            "missing": [],
            "coverage_ratio": 1.0,
        },
        "execution_log_status": execution_log_status,
        "execution_log_digest": execution_log_digest,
    }


def validate_record(record: dict[str, Any]) -> bool:
    required = {
        "strategy", "family_id", "project", "status", "reason_code", "detected",
        "evidence_status", "wall_time_seconds", "subject_invocations",
        "time_to_first_detection_seconds", "plan_digest", "runner_sha256",
        "coverage_certificate", "execution_log_status", "execution_log_digest",
    }
    if not required <= record.keys():
        return False
    certificate = record.get("coverage_certificate", {})
    requested = set(certificate.get("requested", ()))
    covered = set(certificate.get("covered", ()))
    missing = set(certificate.get("missing", ()))
    ratio = certificate.get("coverage_ratio")
    if requested != covered | missing or not isinstance(ratio, (int, float)):
        return False
    expected_ratio = len(covered) / len(requested) if requested else 1.0
    if abs(float(ratio) - expected_ratio) > 1e-12:
        return False
    if record["evidence_status"] == "captured_case_metadata":
        return record["status"] == "ABSTAIN" and not record["detected"] and record["execution_log_digest"] == ""
    if not isinstance(record["execution_log_digest"], str) or len(record["execution_log_digest"]) != 64:
        return False
    return True

"""Conservative, traceable first-pass triage of a locked extension frame."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .benchmark import BenchmarkError


_REASONS = {
    "feature_support_or_non_atomic_request": "feature/support or broad implementation request, not one evidenced historical root defect",
    "documentation_build_typing_or_nonnumerical_interface": "documentation, build, typing, or nonnumerical interface outside the declared component scope",
    "required_platform_unavailable": "reported behavior requires an accelerator or architecture unavailable to the CPU replay host",
    "performance_without_correctness_defect": "performance concern without a stated numerical-correctness defect",
    "upstream_marked_invalid": "upstream issue is labelled invalid; no confirmed root defect in the archived record",
}


def apply_stage1_triage(
    ledger: Mapping[str, Any], frame: Mapping[str, Any], rules: Mapping[str, Any]
) -> dict[str, Any]:
    """Mark only explicitly listed obvious exclusions; everything else stays unresolved."""
    if ledger.get("schema") != "P4-Prospective-Extension-Screening-v1":
        raise BenchmarkError("unsupported source ledger schema")
    if frame.get("schema") != "P4-Prospective-Extension-Frame-v1":
        raise BenchmarkError("unsupported source frame schema")
    if rules.get("schema") != "P4-Extension-Stage1-Triage-Rules-v1":
        raise BenchmarkError("unsupported stage-1 rule schema")
    if len(ledger["rows"]) != len(frame["issues"]) or any(
        row["url"] != issue["url"] for row, issue in zip(ledger["rows"], frame["issues"])
    ):
        raise BenchmarkError("source ledger does not match frozen frame order")
    if any(row["decision"] != "UNSCREENED" for row in ledger["rows"]):
        raise BenchmarkError("source ledger already contains screening decisions")
    exclusions = rules.get("exclude")
    if not isinstance(exclusions, dict) or set(exclusions) != set(_REASONS):
        raise BenchmarkError("stage-1 exclusion categories differ from declared rules")
    lookup = {f"{issue['project']}#{issue['number']}": issue for issue in frame["issues"]}
    selected: dict[str, str] = {}
    for code, identifiers in exclusions.items():
        if not isinstance(identifiers, list):
            raise BenchmarkError("stage-1 exclusion category must contain a list")
        for identifier in identifiers:
            if identifier not in lookup or identifier in selected:
                raise BenchmarkError("stage-1 exclusion ID is unknown or repeated")
            selected[identifier] = code
    output = deepcopy(dict(ledger))
    output["schema"] = "P4-Prospective-Extension-Screening-Pass1-v1"
    output["status"] = "preliminary_topic_triage_no_released_pair_verified"
    output["review_basis"] = rules["basis"]
    for row, issue in zip(output["rows"], frame["issues"]):
        identifier = f"{issue['project']}#{issue['number']}"
        row["issue_title_at_snapshot"] = issue["title"]
        code = selected.get(identifier)
        row["decision"] = "EXCLUDED_STAGE1" if code else "NEEDS_PROVENANCE"
        row["decision_reason"] = _REASONS[code] if code else (
            "not on the explicit stage-1 exclusion list; fix provenance, released pair, root cause, and replay environment remain unverified"
        )
        row["reason_code"] = code
        row["reviewer"] = "internal_stage1" if code else "default_pending_not_individually_admitted"
    output["excluded_stage1_count"] = len(selected)
    output["needs_provenance_count"] = len(output["rows"]) - len(selected)
    output["admitted_count"] = 0
    return output

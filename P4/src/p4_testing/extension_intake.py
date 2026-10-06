"""Mechanical integrity checks for post-freeze extension candidates.

These checks cannot establish semantic root-cause independence or blinding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkError, benchmark_gate, freeze_manifest, load_benchmark


def audit_candidate(
    manifest_path: Path, metadata_path: Path, candidate_path: Path
) -> dict[str, Any]:
    """Fail closed on frozen-manifest drift and literal case/source duplication."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_hash != metadata.get("manifest_sha256"):
        raise BenchmarkError("frozen manifest SHA-256 differs from freeze metadata")
    manifest = load_benchmark(manifest_path)
    frozen = freeze_manifest(manifest)
    if frozen.digest != metadata.get("admitted_manifest_digest"):
        raise BenchmarkError("admitted-case digest differs from freeze metadata")
    gate = benchmark_gate(manifest)
    if (gate.screened_candidates, gate.complete_families, gate.passes) != (
        metadata.get("screened_candidates"),
        metadata.get("admitted_complete"),
        metadata.get("gate_passes"),
    ):
        raise BenchmarkError("frozen benchmark gate differs from freeze metadata")
    required = ("case_id", "project", "defect_family", "upstream_fix")
    if any(not isinstance(candidate.get(key), str) or not candidate[key].strip() for key in required):
        raise BenchmarkError("candidate identity fields are missing")
    if not candidate["upstream_fix"].startswith("https://"):
        raise BenchmarkError("candidate upstream fix must use https")
    issue_url = candidate.get("upstream_issue")
    if issue_url is not None and (
        not isinstance(issue_url, str) or not issue_url.startswith("https://")
    ):
        raise BenchmarkError("candidate upstream issue must use https")
    if any(case.case_id == candidate["case_id"] for case in manifest.cases):
        raise BenchmarkError("candidate case ID duplicates frozen screening inventory")
    if any(case.upstream_url in {candidate["upstream_fix"], issue_url} for case in manifest.cases):
        raise BenchmarkError("candidate upstream URL duplicates frozen screening inventory")
    if any(
        case.project == candidate["project"] and case.defect_family == candidate["defect_family"]
        for case in manifest.cases
    ):
        raise BenchmarkError("candidate project/family pair duplicates frozen screening inventory")
    return {
        "schema": "P4-Extension-Mechanical-Audit-v1",
        "candidate_id": candidate["case_id"],
        "frozen_manifest_sha256": actual_hash,
        "frozen_admitted_digest": frozen.digest,
        "screened_original": gate.screened_candidates,
        "admitted_original": gate.complete_families,
        "original_gate_passes": gate.passes,
        "literal_duplicate_found": False,
        "semantic_root_cause_independence": "requires_human_review",
        "held_out_status": "not_held_out",
    }


def validate_extension_protocol(protocol: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Check the minimum anti-leakage constraints before a new search starts."""
    if protocol.get("schema") != "P4-Prospective-Extension-Protocol-v1":
        raise BenchmarkError("unsupported extension protocol schema")
    if protocol.get("status") != "draft_before_frame_snapshot":
        raise BenchmarkError("protocol status is not the expected pre-snapshot draft")
    if protocol.get("github_query_template") != "repo:{repo} is:issue is:closed created:2023-01-01..2026-06-30 {term}":
        raise BenchmarkError("GitHub search query template differs from the declared rule")
    if protocol.get("sort") != "created-asc":
        raise BenchmarkError("GitHub search sort differs from the declared rule")
    if protocol.get("original_manifest_sha256") != metadata.get("manifest_sha256"):
        raise BenchmarkError("protocol is not bound to the original frozen manifest")
    projects = protocol.get("source_projects")
    terms = protocol.get("search_terms")
    if not isinstance(projects, list) or len(projects) < 5 or len(projects) != len(set(projects)):
        raise BenchmarkError("source projects must be at least five unique entries")
    if not isinstance(terms, list) or not terms or len(terms) != len(set(terms)):
        raise BenchmarkError("search terms must be nonempty and unique")
    cap = protocol.get("records_per_project_term")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise BenchmarkError("search record cap must be positive")
    excluded = protocol.get("development_case_ids_excluded_from_holdout")
    if not isinstance(excluded, list) or "autograd_789" not in excluded:
        raise BenchmarkError("known post-hoc Autograd case must be excluded from holdout")
    if protocol.get("holdout_rule") != "new_snapshot_last_40_percent_per_project_by_ascending_issue_number":
        raise BenchmarkError("holdout partition rule differs from the declared rule")
    if protocol.get("holdout_rounding") != (
        "for each project with n>=2 previously unseen unique issues, reserve the last ceil(0.4*n); if n<2 reserve zero"
    ):
        raise BenchmarkError("holdout rounding rule differs from the declared rule")
    if protocol.get("prior_screened_rule") != (
        "exclude every URL in original frozen.json and the Autograd #789 fix URL from prospective split; retain any matching raw search record"
    ):
        raise BenchmarkError("prior screened records must be excluded from the prospective split")
    if protocol.get("eligibility_decision_before_pair_replay") is not True:
        raise BenchmarkError("eligibility must be recorded before buggy/fixed replay")
    if protocol.get("retain_all_screened_records") is not True:
        raise BenchmarkError("all screened records must be retained")
    if protocol.get("lock_frame_before_replay") is not True:
        raise BenchmarkError("the candidate frame must be locked before replay")

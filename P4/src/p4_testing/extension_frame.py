"""Deterministic, fail-closed construction of a prospective issue-search frame."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

from .benchmark import BenchmarkError, BenchmarkManifest


def query_url(protocol: Mapping[str, Any], project: str, term: str) -> str:
    query = protocol["github_query_template"].format(repo=project, term=term)
    params = {
        "q": query,
        "sort": "created",
        "order": "asc",
        "per_page": protocol["records_per_project_term"],
    }
    return "https://api.github.com/search/issues?" + urlencode(params)


def build_frame(
    protocol: Mapping[str, Any],
    frozen: BenchmarkManifest,
    raw_responses: Mapping[tuple[str, str], tuple[bytes, str]],
    prior_fix_url: str,
) -> dict[str, Any]:
    """Require all configured queries; retain duplicates but split unseen URLs only."""
    pairs = [(p, t) for p in protocol["source_projects"] for t in protocol["search_terms"]]
    if set(raw_responses) != set(pairs):
        raise BenchmarkError("search frame is incomplete or has undeclared queries")
    prior_urls = {case.upstream_url for case in frozen.cases} | {prior_fix_url}
    unique: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    for project, term in pairs:
        raw, retrieved_at = raw_responses[(project, term)]
        try:
            result = json.loads(raw)
        except (ValueError, TypeError) as error:
            raise BenchmarkError("invalid GitHub search JSON") from error
        if not isinstance(result, dict):
            raise BenchmarkError("GitHub search response must be an object")
        items = result.get("items")
        total = result.get("total_count")
        if result.get("incomplete_results") is not False or not isinstance(items, list):
            raise BenchmarkError("GitHub search response is incomplete")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise BenchmarkError("invalid GitHub search total_count")
        cap = protocol["records_per_project_term"]
        if len(items) != min(total, cap):
            raise BenchmarkError("GitHub search returned an unexpected page length")
        previous_created = ""
        for item in items:
            if not isinstance(item, dict):
                raise BenchmarkError("GitHub search item must be an object")
            url = item.get("html_url")
            number = item.get("number")
            created = item.get("created_at")
            title = item.get("title")
            if (not isinstance(url, str)
                    or not url.startswith(f"https://github.com/{project}/issues/")
                    or not isinstance(number, int) or isinstance(number, bool)
                    or url != f"https://github.com/{project}/issues/{number}"
                    or not isinstance(created, str)
                    or not "2023-01-01" <= created[:10] <= "2026-06-30"
                    or not isinstance(title, str)
                    or "pull_request" in item):
                raise BenchmarkError("search item violates repository, date, or issue-only rule")
            if created < previous_created:
                raise BenchmarkError("GitHub search items are not in created-ascending order")
            previous_created = created
            if url not in unique:
                unique[url] = {
                    "url": url, "project": project, "number": number,
                    "title": title, "created_at": created, "matched_terms": [term],
                }
            elif term not in unique[url]["matched_terms"]:
                unique[url]["matched_terms"].append(term)
        queries.append({
            "project": project, "term": term, "url": query_url(protocol, project, term),
            "retrieved_at_utc": retrieved_at,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "total_count": total, "selected_count": len(items),
        })
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in unique.values():
        if item["url"] in prior_urls:
            item["partition"] = "prior_screened_development_only"
        else:
            by_project[item["project"]].append(item)
    for project, items in by_project.items():
        ordered = sorted(items, key=lambda item: item["number"])
        holdout_count = math.ceil(0.4 * len(ordered)) if len(ordered) >= 2 else 0
        for index, item in enumerate(ordered):
            item["partition"] = "prospective_holdout" if index >= len(ordered) - holdout_count else "prospective_development"
    return {
        "schema": "P4-Prospective-Extension-Frame-v1",
        "frame_status": "local_search_snapshot_not_independent_holdout",
        "queries": queries,
        "issues": list(unique.values()),
        "unique_issue_count": len(unique),
        "prior_screened_count": sum(x["partition"] == "prior_screened_development_only" for x in unique.values()),
        "prospective_development_count": sum(x["partition"] == "prospective_development" for x in unique.values()),
        "prospective_holdout_count": sum(x["partition"] == "prospective_holdout" for x in unique.values()),
        "eligibility_decisions_made": 0,
        "buggy_fixed_replays_made": 0,
    }


def verify_locked_frame(
    zip_path: Path, lock_path: Path, manifest_path: Path, candidate_path: Path
) -> dict[str, Any]:
    """Rebuild a local frame from its archived raw responses and compare hashes."""
    from .benchmark import load_benchmark
    from .extension_intake import validate_extension_protocol

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "P4-Prospective-Extension-Local-Lock-v1":
        raise BenchmarkError("unsupported frame lock schema")
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != lock.get("zip_sha256"):
        raise BenchmarkError("frame ZIP SHA-256 differs from lock")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise BenchmarkError("frame ZIP has duplicate or corrupt entries")
        protocol_bytes = archive.read("protocol.json")
        metadata_bytes = archive.read("frozen_metadata.json")
        frame = json.loads(archive.read("frame.json"))
        protocol = json.loads(protocol_bytes)
        metadata = json.loads(metadata_bytes)
        validate_extension_protocol(protocol, metadata)
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if protocol_hash != lock.get("protocol_sha256") or protocol_hash != frame.get("protocol_sha256"):
            raise BenchmarkError("protocol SHA-256 differs from frame lock")
        if manifest_hash != metadata.get("manifest_sha256") or manifest_hash != lock.get("original_manifest_sha256"):
            raise BenchmarkError("original frozen manifest SHA-256 differs from frame lock")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        responses = {}
        for index, query in enumerate(frame["queries"]):
            raw = archive.read(f"raw/query_{index:02d}.json")
            if hashlib.sha256(raw).hexdigest() != query.get("raw_sha256"):
                raise BenchmarkError("raw search response SHA-256 differs from frame")
            responses[(query["project"], query["term"])] = (raw, query["retrieved_at_utc"])
        expected_names = {"protocol.json", "frozen_metadata.json", "frame.json"} | {
            f"raw/query_{index:02d}.json" for index in range(len(frame["queries"]))
        }
        if set(names) != expected_names:
            raise BenchmarkError("frame ZIP has missing or undeclared entries")
    rebuilt = build_frame(protocol, load_benchmark(manifest_path), responses, candidate["upstream_fix"])
    for key, value in rebuilt.items():
        if frame.get(key) != value:
            raise BenchmarkError(f"frame differs from archived raw responses: {key}")
    for key in ("query_count", "unique_issue_count", "prior_screened_count",
                "prospective_development_count", "prospective_holdout_count"):
        expected = len(rebuilt["queries"]) if key == "query_count" else rebuilt[key]
        if lock.get(key) != expected:
            raise BenchmarkError(f"frame lock count differs: {key}")
    return lock


def initialize_screening_ledger(frame: Mapping[str, Any], zip_sha256: str) -> dict[str, Any]:
    """Preserve every framed issue as pending; no eligibility decision is inferred."""
    if frame.get("schema") != "P4-Prospective-Extension-Frame-v1":
        raise BenchmarkError("unsupported frame schema for screening ledger")
    if len(frame["issues"]) != frame["unique_issue_count"]:
        raise BenchmarkError("frame issue count differs from unique count")
    return {
        "schema": "P4-Prospective-Extension-Screening-v1",
        "source_zip_sha256": zip_sha256,
        "status": "all_records_pending_pre_replay_screening",
        "rows": [
            {
                "url": issue["url"],
                "project": issue["project"],
                "issue_number": issue["number"],
                "partition": issue["partition"],
                "decision": "UNSCREENED",
                "root_defect_family": None,
                "affected_release": None,
                "fixed_release": None,
                "decision_reason": None,
                "reviewer": None,
                "reviewed_at_utc": None,
            }
            for issue in frame["issues"]
        ],
    }

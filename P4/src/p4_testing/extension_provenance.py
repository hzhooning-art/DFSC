"""Ordered, hash-bound provenance capture for unresolved development issues."""

from __future__ import annotations

import hashlib
import io
import base64
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .benchmark import BenchmarkError


def first_pending_development(screening: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    if screening.get("schema") != "P4-Prospective-Extension-Screening-Pass1-v1":
        raise BenchmarkError("unsupported screening ledger for provenance capture")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise BenchmarkError("provenance batch size must be positive")
    rows = [row for row in screening["rows"] if row["partition"] == "prospective_development"
            and row["decision"] == "NEEDS_PROVENANCE"]
    return rows[:count]


def verify_provenance_capture(
    zip_path: Path, lock_path: Path, screening_path: Path, *, batch_start: int = 0
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "P4-Extension-Provenance-Capture-Lock-v1":
        raise BenchmarkError("unsupported provenance capture lock")
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != lock.get("zip_sha256"):
        raise BenchmarkError("provenance ZIP hash differs from lock")
    if hashlib.sha256(screening_path.read_bytes()).hexdigest() != lock.get("source_screening_sha256"):
        raise BenchmarkError("screening ledger hash differs from provenance lock")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if isinstance(batch_start, bool) or batch_start not in {0, 5, 10, 15}:
        raise BenchmarkError("unsupported provenance batch start")
    expected = [row["url"] for row in first_pending_development(screening, batch_start + 5)
                [batch_start:batch_start + 5]]
    if len(expected) != 5 or lock.get("batch_start", 0) != batch_start or lock.get("issue_urls") != expected:
        raise BenchmarkError("provenance batch differs from its fixed five-issue frame slice")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise BenchmarkError("provenance ZIP has duplicate or damaged entries")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("batch_start", 0) != batch_start or manifest.get("issue_urls") != expected:
            raise BenchmarkError("provenance manifest issue order differs from screening ledger")
        expected_names = {"manifest.json"}
        for item in manifest["responses"]:
            name = item["archive_path"]
            expected_names.add(name)
            if hashlib.sha256(archive.read(name)).hexdigest() != item["sha256"]:
                raise BenchmarkError("raw provenance response hash differs from manifest")
        if set(names) != expected_names or lock.get("response_count") != len(manifest["responses"]):
            raise BenchmarkError("provenance ZIP response inventory differs from lock")
    return lock


def verify_provenance_decisions(
    decisions_path: Path, zip_path: Path, lock_path: Path, screening_path: Path,
    *, batch_start: int = 0,
) -> dict[str, Any]:
    """Validate a batch decision overlay without rewriting the frozen frame or pass 1."""
    lock = verify_provenance_capture(zip_path, lock_path, screening_path,
                                     batch_start=batch_start)
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions.get("schema") != f"P4-Extension-Provenance-Batch{1 + batch_start // 5}-Decisions-v1":
        raise BenchmarkError("unsupported provenance decision overlay")
    if decisions.get("source_capture_sha256") != lock["zip_sha256"]:
        raise BenchmarkError("decision overlay is not bound to provenance capture")
    if decisions.get("source_screening_sha256") != lock["source_screening_sha256"]:
        raise BenchmarkError("decision overlay is not bound to pass-1 screening")
    rows = decisions.get("rows")
    if not isinstance(rows, list) or [row.get("url") for row in rows] != lock["issue_urls"]:
        raise BenchmarkError("decision overlay must retain its five issues in frame order")
    allowed = {"EXCLUDED_AFTER_PROVENANCE", "NEEDS_SCOPE_AND_RELEASE_PROVENANCE"}
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        captured = {(item["issue_url"], item["kind"]): item["archive_path"]
                    for item in manifest["responses"]}
    for row in rows:
        if row.get("decision") not in allowed or not row.get("reason_code") or not row.get("rationale"):
            raise BenchmarkError("decision row lacks a supported non-admission disposition")
        if row.get("affected_release") is not None or row.get("fixed_release") is not None:
            raise BenchmarkError("issue-only decisions may not assert unverified release pairs")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != {"issue", "comments", "timeline"}:
            raise BenchmarkError("decision row must cite all captured upstream evidence kinds")
        for kind, path in evidence.items():
            if path != captured.get((row["url"], kind)):
                raise BenchmarkError("decision evidence path does not match captured issue")
    return decisions


def verify_autograd_590_followup(
    zip_path: Path, lock_path: Path, batch1_zip: Path, batch1_decisions: Path
) -> dict[str, Any]:
    """Verify immutable PR/release provenance, without claiming cohort eligibility."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "P4-Autograd-590-Followup-Lock-v1":
        raise BenchmarkError("unsupported Autograd follow-up lock")
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != lock.get("zip_sha256"):
        raise BenchmarkError("Autograd follow-up ZIP hash differs from lock")
    for path, key in ((batch1_zip, "prior_batch1_sha256"),
                      (batch1_decisions, "prior_decisions_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock.get(key):
            raise BenchmarkError("Autograd follow-up is not bound to prior evidence")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise BenchmarkError("Autograd follow-up ZIP is duplicated or damaged")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("issue_url") != lock.get("issue_url") or lock.get("source_count") != 6:
            raise BenchmarkError("Autograd follow-up identity or inventory differs")
        if manifest.get("prior_batch1_sha256") != lock["prior_batch1_sha256"] or (
            manifest.get("prior_decisions_sha256") != lock["prior_decisions_sha256"]
        ):
            raise BenchmarkError("Autograd follow-up manifest binding differs")
        sources = manifest["sources"]
        if {item["kind"] for item in sources} != {
            "pr", "pr_files", "pypi_1_5", "pypi_1_6", "wheel_1_5", "wheel_1_6"
        } or len(sources) != 6:
            raise BenchmarkError("Autograd follow-up missing a required source")
        if set(names) != {"manifest.json", *(item["archive_path"] for item in sources)}:
            raise BenchmarkError("Autograd follow-up contains unexpected entries")
        raw = {}
        for item in sources:
            data = archive.read(item["archive_path"])
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise BenchmarkError("Autograd follow-up source hash differs")
            raw[item["kind"]] = data
        pr = json.loads(raw["pr"])
        if pr.get("number") != 591 or not pr.get("merged"):
            raise BenchmarkError("Autograd PR identity/merge state differs")
        if {item["filename"] for item in json.loads(raw["pr_files"])} != {
            "autograd/numpy/numpy_boxes.py", "autograd/numpy/numpy_vspaces.py"
        }:
            raise BenchmarkError("Autograd PR file scope differs")
        for version in ("1_5", "1_6"):
            metadata = json.loads(raw[f"pypi_{version}"])
            expected_version = version.replace("_", ".")
            if metadata["info"]["version"] != expected_version:
                raise BenchmarkError("Autograd PyPI version differs")
            wheel = next(item for item in metadata["urls"] if item["packagetype"] == "bdist_wheel")
            if hashlib.sha256(raw[f"wheel_{version}"]).hexdigest() != wheel["digests"]["sha256"]:
                raise BenchmarkError("Autograd official wheel digest differs")
        old = zipfile.ZipFile(io.BytesIO(raw["wheel_1_5"]))
        new = zipfile.ZipFile(io.BytesIO(raw["wheel_1_6"]))
        for name in ("autograd/numpy/numpy_boxes.py", "autograd/numpy/numpy_vspaces.py"):
            if b"np.float128" in old.read(name) or b"np.complex256" in old.read(name):
                raise BenchmarkError("old Autograd wheel already has expanded dtype registration")
            if b"np.float128" not in new.read(name) or b"np.complex256" not in new.read(name):
                raise BenchmarkError("new Autograd wheel lacks expanded dtype registration")
    return lock


def verify_autograd_590_decision(
    decision_path: Path, zip_path: Path, lock_path: Path,
    batch1_zip: Path, batch1_decisions: Path,
) -> dict[str, Any]:
    lock = verify_autograd_590_followup(zip_path, lock_path,
                                        batch1_zip, batch1_decisions)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("schema") != "P4-Autograd-590-Followup-Decision-v1" or (
        decision.get("issue_url") != lock["issue_url"]
    ):
        raise BenchmarkError("Autograd #590 decision identity differs")
    if decision.get("followup_zip_sha256") != lock["zip_sha256"] or (
        decision.get("prior_batch1_decisions_sha256") != lock["prior_decisions_sha256"]
    ):
        raise BenchmarkError("Autograd #590 decision is not bound to evidence")
    if decision.get("disposition") != "EXCLUDED_AFTER_SCOPE_REVIEW" or (
        decision.get("pre_change_release"), decision.get("post_change_release")
    ) != ("1.5", "1.6"):
        raise BenchmarkError("Autograd #590 disposition or release chronology differs")
    if decision.get("historical_defect_family_admitted") is not False or (
        decision.get("buggy_fixed_runner_executed") is not False
    ):
        raise BenchmarkError("Autograd #590 falsely claims an admitted or executed defect")
    return decision


def verify_pytorch_91631_followup(
    zip_path: Path, lock_path: Path, batch3_zip: Path, batch3_decisions: Path
) -> dict[str, Any]:
    """Verify the upstream fix and two released tag sources, not runtime behavior."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "P4-PyTorch-91631-Followup-Lock-v1" or (
        lock.get("issue_url") != "https://github.com/pytorch/pytorch/issues/91631"
    ):
        raise BenchmarkError("unsupported PyTorch #91631 follow-up identity")
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != lock.get("zip_sha256"):
        raise BenchmarkError("PyTorch #91631 follow-up ZIP hash differs")
    for path, key in ((batch3_zip, "prior_batch3_sha256"),
                      (batch3_decisions, "prior_decisions_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock.get(key):
            raise BenchmarkError("PyTorch #91631 follow-up prior binding differs")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise BenchmarkError("PyTorch #91631 follow-up ZIP is damaged or duplicated")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("issue_url") != lock["issue_url"] or (
            manifest.get("fix_commit") != lock.get("fix_commit")
        ) or manifest.get("prior_batch3_sha256") != lock["prior_batch3_sha256"] or (
            manifest.get("prior_decisions_sha256") != lock["prior_decisions_sha256"]
        ):
            raise BenchmarkError("PyTorch #91631 follow-up manifest binding differs")
        sources = manifest["responses"]
        expected = {"fix_commit", "pull_request", "pull_request_files",
                    "source_v1_13_1", "source_v2_0_0"}
        if len(sources) != 5 or {item["kind"] for item in sources} != expected or (
            lock.get("response_count") != 5
        ) or set(names) != {"manifest.json", *(item["archive_path"] for item in sources)}:
            raise BenchmarkError("PyTorch #91631 follow-up source inventory differs")
        raw = {}
        for item in sources:
            data = archive.read(item["archive_path"])
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise BenchmarkError("PyTorch #91631 follow-up raw source hash differs")
            raw[item["kind"]] = json.loads(data)
        commit = raw["fix_commit"]
        source_path = "aten/src/ATen/templates/TensorBody.h"
        if commit.get("sha") != lock["fix_commit"] or (
            "Fixes https://github.com/pytorch/pytorch/issues/91631"
            not in commit["commit"]["message"]
        ) or {item["filename"] for item in commit["files"]} != {source_path}:
            raise BenchmarkError("PyTorch #91631 closing commit identity differs")
        if raw["pull_request"].get("number") != 91695 or (
            {item["filename"] for item in raw["pull_request_files"]} != {source_path}
        ):
            raise BenchmarkError("PyTorch #91631 linked PR scope differs")
        old = raw["source_v1_13_1"]
        new = raw["source_v2_0_0"]
        if old["path"] != source_path or new["path"] != source_path or (
            old["encoding"] != "base64" or new["encoding"] != "base64"
        ):
            raise BenchmarkError("PyTorch #91631 tag source identity differs")
        old_source = base64.b64decode(old["content"])
        new_source = base64.b64decode(new["content"])
        if b"if (this->is_sparse())" not in old_source or (
            b"case at::kSparseCsr:" in old_source
        ) or b"case at::kSparseCsr:" not in new_source or (
            b"case at::kSparseCsc:" not in new_source
        ):
            raise BenchmarkError("PyTorch #91631 released source change differs")
    return lock


def verify_pytorch_91631_decision(
    decision_path: Path, zip_path: Path, lock_path: Path,
    batch3_zip: Path, batch3_decisions: Path,
) -> dict[str, Any]:
    lock = verify_pytorch_91631_followup(zip_path, lock_path,
                                         batch3_zip, batch3_decisions)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("schema") != "P4-PyTorch-91631-Followup-Decision-v1" or (
        decision.get("issue_url") != lock["issue_url"]
    ) or decision.get("followup_zip_sha256") != lock["zip_sha256"] or (
        decision.get("prior_batch3_decisions_sha256") != lock["prior_decisions_sha256"]
    ):
        raise BenchmarkError("PyTorch #91631 decision binding differs")
    if decision.get("disposition") != "RELEASE_PROVENANCE_VERIFIED_REPLAY_PENDING" or (
        decision.get("affected_release"), decision.get("verified_fixed_tag")
    ) != ("v1.13.1", "v2.0.0"):
        raise BenchmarkError("PyTorch #91631 decision overstates release provenance")
    if decision.get("historical_defect_family_admitted") is not False or (
        decision.get("buggy_fixed_runner_executed") is not False
    ):
        raise BenchmarkError("PyTorch #91631 decision falsely claims execution/admission")
    return decision

"""Strict, auditable manifests for historical buggy/fixed benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class BenchmarkError(ValueError):
    """Raised when benchmark evidence cannot be admitted or frozen."""


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    project: str
    upstream_url: str
    defect_family: str
    observable: str
    runner_path: str
    environment: Mapping[str, Any]
    buggy_version: str | None
    fixed_version: str | None
    admission_state: str
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True)
class BenchmarkManifest:
    schema: str
    cases: tuple[BenchmarkCase, ...]

    @property
    def admitted_cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(case for case in self.cases if case.admission_state == "admitted_complete")

    def with_duplicate_variant(self, case_id: str) -> "BenchmarkManifest":
        try:
            source = next(case for case in self.cases if case.case_id == case_id)
        except StopIteration as error:
            raise BenchmarkError(f"unknown case: {case_id}") from error
        duplicate = replace(source, case_id=f"{case_id}__duplicate")
        return replace(self, cases=(*self.cases, duplicate))


@dataclass(frozen=True)
class FrozenManifest:
    schema: str
    cases: tuple[BenchmarkCase, ...]
    digest: str


@dataclass(frozen=True)
class ProjectSplit:
    held_out_project: str
    training: tuple[BenchmarkCase, ...]
    held_out: tuple[BenchmarkCase, ...]


@dataclass(frozen=True)
class BenchmarkGate:
    screened_candidates: int
    complete_families: int
    projects: int
    projects_with_two_families: int
    resolved: bool
    passes: bool


@dataclass(frozen=True)
class CapturedCase:
    case_id: str
    project: str
    defect_family: str
    upstream_url: str
    runner_path: Path
    capture_path: Path
    runner_sha256: str
    detected_by: tuple[str, ...]


_REQUIRED_CASE_FIELDS = {
    "case_id",
    "project",
    "upstream_url",
    "defect_family",
    "observable",
    "runner_path",
    "environment",
    "buggy_version",
    "fixed_version",
    "admission_state",
    "exclusion_reason",
}
_ADMISSION_STATES = {"admitted_complete", "candidate", "excluded"}


def validate_case(case: BenchmarkCase) -> None:
    for field in (
        "case_id",
        "project",
        "upstream_url",
        "defect_family",
        "observable",
        "runner_path",
    ):
        value = getattr(case, field)
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkError(f"{field} must be a non-empty string")
    if not case.upstream_url.startswith("https://"):
        raise BenchmarkError("upstream_url must use https")
    if not case.environment:
        raise BenchmarkError("environment must be recorded")
    if case.admission_state not in _ADMISSION_STATES:
        raise BenchmarkError(f"invalid admission state: {case.admission_state}")
    if case.admission_state == "admitted_complete":
        if not case.buggy_version or not case.fixed_version:
            raise BenchmarkError("admitted complete cases require both versions")
        if case.exclusion_reason is not None:
            raise BenchmarkError("admitted complete cases cannot have an exclusion reason")
    elif not case.exclusion_reason:
        raise BenchmarkError("non-admitted cases require an exclusion reason")


def _parse_case(raw: Mapping[str, Any]) -> BenchmarkCase:
    unknown = set(raw) - _REQUIRED_CASE_FIELDS
    missing = _REQUIRED_CASE_FIELDS - set(raw)
    if unknown:
        raise BenchmarkError(f"unknown case fields: {sorted(unknown)}")
    if missing:
        raise BenchmarkError(f"missing case fields: {sorted(missing)}")
    if not isinstance(raw["environment"], Mapping):
        raise BenchmarkError("environment must be a mapping")
    case = BenchmarkCase(**raw)
    validate_case(case)
    return case


def load_benchmark(path: Path) -> BenchmarkManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot load benchmark {path}: {error}") from error
    if not isinstance(raw, Mapping) or set(raw) != {"schema", "cases"}:
        raise BenchmarkError("benchmark requires exactly schema and cases")
    if raw["schema"] != "P4-Historical-Benchmark-v1":
        raise BenchmarkError("unsupported benchmark schema")
    if not isinstance(raw["cases"], list):
        raise BenchmarkError("cases must be a list")
    cases = tuple(_parse_case(item) for item in raw["cases"] if isinstance(item, Mapping))
    if len(cases) != len(raw["cases"]):
        raise BenchmarkError("every case must be a mapping")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise BenchmarkError("case ids must be unique")
    return BenchmarkManifest(raw["schema"], cases)


def _case_payload(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "project": case.project,
        "upstream_url": case.upstream_url,
        "defect_family": case.defect_family,
        "observable": case.observable,
        "runner_path": case.runner_path,
        "environment": dict(case.environment),
        "buggy_version": case.buggy_version,
        "fixed_version": case.fixed_version,
        "admission_state": case.admission_state,
        "exclusion_reason": case.exclusion_reason,
    }


def freeze_manifest(manifest: BenchmarkManifest) -> FrozenManifest:
    admitted = tuple(case for case in manifest.cases if case.admission_state == "admitted_complete")
    families = [case.defect_family for case in admitted]
    if len(families) != len(set(families)):
        raise BenchmarkError("duplicate defect family among admitted cases")
    if not admitted:
        raise BenchmarkError("cannot freeze an empty benchmark")
    payload = {
        "schema": manifest.schema,
        "cases": [_case_payload(case) for case in sorted(admitted, key=lambda item: item.case_id)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return FrozenManifest(manifest.schema, admitted, digest)


def benchmark_gate(manifest: BenchmarkManifest) -> BenchmarkGate:
    admitted = manifest.admitted_cases
    families = {case.defect_family for case in admitted}
    projects = {case.project for case in admitted}
    families_by_project = {
        project: {case.defect_family for case in admitted if case.project == project}
        for project in projects
    }
    complete_families = len(families)
    project_count = len(projects)
    projects_with_two = sum(len(items) >= 2 for items in families_by_project.values())
    resolved = len(manifest.cases) >= 30 and all(
        case.admission_state in {"admitted_complete", "excluded"} for case in manifest.cases
    )
    passes = (
        complete_families >= 20
        and project_count >= 5
        and projects_with_two >= 4
    )
    return BenchmarkGate(
        screened_candidates=len(manifest.cases),
        complete_families=complete_families,
        projects=project_count,
        projects_with_two_families=projects_with_two,
        resolved=resolved,
        passes=passes,
    )


def leave_one_project_out(manifest: FrozenManifest) -> tuple[ProjectSplit, ...]:
    projects = sorted({case.project for case in manifest.cases})
    return tuple(
        ProjectSplit(
            project,
            tuple(case for case in manifest.cases if case.project != project),
            tuple(case for case in manifest.cases if case.project == project),
        )
        for project in projects
    )


def load_case(path: Path) -> CapturedCase:
    """Load one migrated capture adapter using paths relative to the P4 root."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot load captured case {path}: {error}") from error
    required = {
        "schema",
        "case_id",
        "project",
        "defect_family",
        "upstream_url",
        "runner_path",
        "capture_path",
        "runner_sha256",
        "detected_by",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise BenchmarkError("captured case has missing or unknown fields")
    if raw["schema"] != "P4-Captured-Case-v1":
        raise BenchmarkError("unsupported captured-case schema")
    detected_by = raw["detected_by"]
    if not isinstance(detected_by, list) or not detected_by or not all(
        isinstance(item, str) and item for item in detected_by
    ):
        raise BenchmarkError("detected_by must be a non-empty string list")
    p4_root = path.resolve().parents[2]
    runner_path = (p4_root / raw["runner_path"]).resolve()
    capture_path = (p4_root / raw["capture_path"]).resolve()
    if not runner_path.is_relative_to(p4_root) or not capture_path.is_relative_to(p4_root):
        raise BenchmarkError("captured-case paths must remain inside P4")
    digest = raw["runner_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise BenchmarkError("runner_sha256 must contain 64 hexadecimal characters")
    return CapturedCase(
        case_id=raw["case_id"],
        project=raw["project"],
        defect_family=raw["defect_family"],
        upstream_url=raw["upstream_url"],
        runner_path=runner_path,
        capture_path=capture_path,
        runner_sha256=digest.lower(),
        detected_by=tuple(detected_by),
    )

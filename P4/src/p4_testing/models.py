"""Immutable domain models shared by the reconstructed P4 study."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NON_EVALUABLE = "non_evaluable"
    INVALID_CASE = "invalid_case"
    REPRODUCTION_MISMATCH = "reproduction_mismatch"


class Layer(str, Enum):
    REFERENCE = "reference"
    PROPERTY = "property"
    EXECUTION = "execution"
    REPLAY = "replay"


_FORBIDDEN_RELATION_TOKENS = (
    "buggy",
    "fixed",
    "expected_role",
    "package_version",
)


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class TestSpec:
    test_id: str
    layer: Layer
    relation: str
    parameters: Mapping[str, Any]
    max_invocations: int
    marginal_mechanisms: tuple[str, ...] = ()
    estimated_cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.test_id.strip():
            raise ValueError("test_id must be non-empty")
        if self.max_invocations <= 0:
            raise ValueError("max_invocations must be positive")
        normalized = self.relation.casefold()
        for token in _FORBIDDEN_RELATION_TOKENS:
            if token in normalized:
                raise ValueError(f"relation contains forbidden token: {token}")
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))


@dataclass(frozen=True)
class TestPlan:
    schema: str
    contract_id: str
    seed: int
    tests: tuple[TestSpec, ...]
    layers: tuple[Layer, ...]
    max_invocations: int
    wall_seconds: float
    digest: str
    risk_mechanisms: tuple[str, ...] = ()
    status: str = "READY"
    reason_code: str | None = None
    coverage_certificate: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coverage_certificate", _immutable_mapping(self.coverage_certificate))


@dataclass(frozen=True)
class CaseManifest:
    case_id: str
    defect_family: str
    project: str
    upstream_url: str
    runner_path: str
    buggy: Mapping[str, Any]
    fixed: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "buggy", _immutable_mapping(self.buggy))
        object.__setattr__(self, "fixed", _immutable_mapping(self.fixed))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class TestObservation:
    test_id: str
    outcome: Outcome
    invocations: int
    elapsed_seconds: float
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.invocations < 0:
            raise ValueError("invocations cannot be negative")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")
        object.__setattr__(self, "details", _immutable_mapping(self.details))


@dataclass(frozen=True)
class PairResult:
    case_id: str
    defect_family: str
    project: str
    status: str
    detected: bool
    buggy: TestObservation
    fixed: TestObservation
    runner_sha256: str
    runner_hash_matches: bool
    plan_digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

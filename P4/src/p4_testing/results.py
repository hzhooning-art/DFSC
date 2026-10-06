"""Pair classification and canonical serialization for P4 evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .models import Outcome, PairResult, TestObservation


@dataclass(frozen=True)
class PairOutcome:
    status: str
    detected: bool


@dataclass(frozen=True)
class ConformanceResult:
    """Tri-state decision for a compiled evidence plan."""

    status: str
    reason_code: str | None
    observed_test_ids: tuple[str, ...]

    @classmethod
    def from_observations(
        cls,
        observations: list[Mapping[str, Any]],
        *,
        mandatory: tuple[str, ...],
    ) -> "ConformanceResult":
        observed = tuple(str(item.get("test_id", "")) for item in observations)
        missing = [item for item in mandatory if item not in observed]
        if missing:
            return cls("ABSTAIN", "MISSING_MANDATORY_EVIDENCE", observed)
        if any(item.get("status") == "fail" for item in observations):
            return cls("NONCONFORMANT", "MANDATORY_GATE_FAILED", observed)
        if any(item.get("status") in {"non_evaluable", "invalid_case"} for item in observations):
            return cls("ABSTAIN", "NON_EVALUABLE_EVIDENCE", observed)
        return cls("CONFORMANT", None, observed)


def classify_pair(
    buggy: TestObservation,
    fixed: TestObservation,
) -> PairOutcome:
    outcomes = {buggy.outcome, fixed.outcome}
    if Outcome.REPRODUCTION_MISMATCH in outcomes:
        return PairOutcome("reproduction_mismatch", False)
    if Outcome.INVALID_CASE in outcomes:
        return PairOutcome("invalid_case", False)
    if Outcome.NON_EVALUABLE in outcomes:
        return PairOutcome("non_evaluable", False)
    if buggy.outcome is Outcome.FAIL and fixed.outcome is Outcome.PASS:
        return PairOutcome("detected", True)
    if fixed.outcome is Outcome.FAIL:
        return PairOutcome("fixed_regression", False)
    return PairOutcome("missed", False)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Outcome):
        return value.value
    return value


def _observation_payload(observation: TestObservation) -> dict[str, Any]:
    return {
        "test_id": observation.test_id,
        "outcome": observation.outcome.value,
        "invocations": observation.invocations,
        "elapsed_seconds": observation.elapsed_seconds,
        "details": _json_value(observation.details),
    }


def canonical_result_bytes(result: PairResult) -> bytes:
    payload = {
        "case_id": result.case_id,
        "defect_family": result.defect_family,
        "project": result.project,
        "status": result.status,
        "detected": result.detected,
        "buggy": _observation_payload(result.buggy),
        "fixed": _observation_payload(result.fixed),
        "runner_sha256": result.runner_sha256,
        "runner_hash_matches": result.runner_hash_matches,
        "plan_digest": result.plan_digest,
        "metadata": _json_value(result.metadata),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

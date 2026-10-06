"""Strict schema for fresh, independently executed baseline strategy runs.

Historical ``detected_by`` metadata is intentionally incompatible with this
schema. A row is admissible only when timing, invocation count, runner and
environment hashes, and an execution provenance label are present.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class IndependentStrategyRun:
    case_id: str
    strategy: str
    status: str
    detected: bool
    elapsed_seconds: float
    subject_invocations: int
    runner_sha256: str
    environment_sha256: str
    provenance: str
    preregistration_sha256: str


def run_digest(run: IndependentStrategyRun) -> str:
    payload = {"case_id": run.case_id, "strategy": run.strategy, "status": run.status,
               "detected": run.detected, "elapsed_seconds": run.elapsed_seconds,
               "subject_invocations": run.subject_invocations,
               "runner_sha256": run.runner_sha256, "environment_sha256": run.environment_sha256,
               "provenance": run.provenance, "preregistration_sha256": run.preregistration_sha256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_independent_run(run: Mapping[str, Any]) -> bool:
    required = {"case_id", "strategy", "status", "detected", "elapsed_seconds",
                "subject_invocations", "runner_sha256", "environment_sha256", "provenance", "preregistration_sha256"}
    if set(run) != required:
        return False
    if not isinstance(run["case_id"], str) or not run["case_id"]:
        return False
    if not isinstance(run["strategy"], str) or not run["strategy"]:
        return False
    if run["status"] not in {"detected", "missed", "abstain"} or not isinstance(run["detected"], bool):
        return False
    if run["status"] == "detected" and not run["detected"]:
        return False
    if run["status"] in {"missed", "abstain"} and run["detected"]:
        return False
    if (not isinstance(run["elapsed_seconds"], (int, float))
            or not math.isfinite(float(run["elapsed_seconds"]))
            or run["elapsed_seconds"] < 0):
        return False
    if isinstance(run["subject_invocations"], bool) or not isinstance(run["subject_invocations"], int) or run["subject_invocations"] < 0:
        return False
    for key in ("runner_sha256", "environment_sha256"):
        value = run[key]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            return False
    digest = run["preregistration_sha256"]
    return (run["provenance"] == "independent_frozen_execution"
            and isinstance(digest, str) and len(digest) == 64
            and all(ch in "0123456789abcdef" for ch in digest.lower()))

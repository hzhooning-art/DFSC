"""Auditable per-side execution logs for historical pair captures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExecutionRun:
    role: str
    run_index: int
    outcome: str
    environment_sha256: str
    runner_sha256: str
    cost_status: str
    elapsed_seconds: float | None
    subject_invocations: int | None


@dataclass(frozen=True)
class PairExecutionLog:
    case_id: str
    runner_sha256: str
    runner_unchanged_between_sides: bool
    runs: tuple[ExecutionRun, ...]
    evidence_status: str


def execution_log_payload(log: PairExecutionLog) -> dict[str, Any]:
    """Return the canonical, JSON-safe representation used for provenance."""
    return {
        "schema": "P4-Execution-Log-v1",
        "case_id": log.case_id,
        "runner_sha256": log.runner_sha256,
        "runner_unchanged_between_sides": log.runner_unchanged_between_sides,
        "evidence_status": log.evidence_status,
        "runs": [
            {
                "role": run.role,
                "run_index": run.run_index,
                "outcome": run.outcome,
                "environment_sha256": run.environment_sha256,
                "runner_sha256": run.runner_sha256,
                "cost_status": run.cost_status,
                "elapsed_seconds": run.elapsed_seconds,
                "subject_invocations": run.subject_invocations,
            }
            for run in log.runs
        ],
    }


def execution_log_digest(log: PairExecutionLog) -> str:
    encoded = json.dumps(execution_log_payload(log), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_hash(environment: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(environment), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_pair_execution_log(path: Path) -> PairExecutionLog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    runs: list[ExecutionRun] = []
    for role in ("buggy", "fixed"):
        # Early captures used one shared ``legacy_environment`` and did not
        # record a repeat count.  Preserve them as single captured runs while
        # keeping the missing cost fields explicitly unobserved.
        environment = raw.get(f"{role}_environment", raw.get("legacy_environment", {}))
        side = raw[f"{role}_side"]
        clean_runs = int(side.get("clean_runs", 1 if side.get("result") else 0))
        result = side.get("result", {})
        outcome = str(result.get("outcome", "captured"))
        if not outcome or outcome == "None":
            outcome = "captured"
        for run_index in range(clean_runs):
            runs.append(
                ExecutionRun(
                    role=role,
                    run_index=run_index + 1,
                    outcome=outcome,
                    environment_sha256=_environment_hash(environment),
                    runner_sha256=str(raw["runner_sha256"]),
                    cost_status="UNOBSERVED",
                    elapsed_seconds=None,
                    subject_invocations=None,
                )
            )
    return PairExecutionLog(
        case_id=str(raw["case_id"]),
        runner_sha256=str(raw["runner_sha256"]),
        runner_unchanged_between_sides=bool(raw["runner_unchanged_between_sides"]),
        runs=tuple(runs),
        evidence_status="captured_pair_replay",
    )


def validate_execution_log(log: PairExecutionLog) -> bool:
    if not log.runner_unchanged_between_sides or not log.runs:
        return False
    if any(run.runner_sha256 != log.runner_sha256 for run in log.runs):
        return False
    roles = {run.role for run in log.runs}
    if roles != {"buggy", "fixed"}:
        return False
    return all(run.cost_status in {"OBSERVED", "UNOBSERVED"} for run in log.runs)

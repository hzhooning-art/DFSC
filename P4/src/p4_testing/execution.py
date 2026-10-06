"""Execute every selected relation through an explicitly supplied command registry.

The registry is trusted local configuration, never a command extracted from a
component contract. A worker is responsible for its own numerical oracle; this
module checks execution completeness and preserves the raw process evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .compiler import Budget, canonical_plan_bytes, compile_plan
from .contracts import ComponentContract, RiskProfile
from .models import TestPlan, TestSpec


OBSERVATION_SCHEMA = "P4-Relation-Observation-v1"
REPORT_SCHEMA = "P4-Plan-Execution-v1"


def verify_plan_contract(plan: TestPlan, contract: ComponentContract) -> None:
    """Reject a plan whose declarations no longer match its source contract."""

    if not plan.risk_mechanisms:
        raise ValueError("execution requires a risk-aware plan")
    profile = RiskProfile.from_dict({"mechanisms": list(plan.risk_mechanisms)})
    rebuilt = compile_plan(
        contract,
        profile,
        Budget(plan.max_invocations, plan.wall_seconds),
        seed=plan.seed,
    )
    if canonical_plan_bytes(rebuilt) != canonical_plan_bytes(plan):
        raise ValueError("plan does not match current contract and risk profile")


def load_registry(path: Path) -> dict[str, tuple[str, ...]]:
    """Load a local registry of argv vectors; shell interpretation is forbidden."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "P4-Execution-Registry-v1":
        raise ValueError("unsupported execution registry")
    commands = raw.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("registry commands must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for test_id, command in commands.items():
        if not isinstance(test_id, str) or not test_id:
            raise ValueError("registry test IDs must be non-empty strings")
        if not isinstance(command, list) or not command or any(
            not isinstance(arg, str) or not arg for arg in command
        ):
            raise ValueError(f"registry command for {test_id} must be an argv array")
        result[test_id] = tuple(command)
    return result


def _expand(command: tuple[str, ...], test: TestSpec, root: Path) -> list[str]:
    tokens = {"{python}": sys.executable, "{p4_root}": str(root), "{test_id}": test.test_id}
    return [
        token.replace("{python}", tokens["{python}"])
        .replace("{p4_root}", tokens["{p4_root}"])
        .replace("{test_id}", tokens["{test_id}"])
        for token in command
    ]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _process_record(test: TestSpec, command: tuple[str, ...] | None, deadline: float, root: Path) -> dict[str, Any]:
    base: dict[str, Any] = {"test_id": test.test_id, "outcome": "ABSTAIN", "subject_invocations": 0}
    if command is None:
        return {**base, "reason": "MISSING_EXECUTOR"}
    argv = _expand(command, test, root)
    command_hash = _sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8"))
    files = {
        str(Path(arg).resolve()): _sha256(Path(arg).read_bytes())
        for arg in argv if Path(arg).is_file()
    }
    base.update({"command_argv": argv, "command_sha256": command_hash, "command_file_sha256": files})
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return {**base, "reason": "WALL_BUDGET_EXHAUSTED"}
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            argv, cwd=root, capture_output=True, timeout=remaining, check=False, shell=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout_bytes, stderr_bytes = error.stdout or b"", error.stderr or b""
        return {**base, "reason": "TIMEOUT", "elapsed_seconds": time.perf_counter() - start,
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "stdout_sha256": _sha256(stdout_bytes), "stderr_sha256": _sha256(stderr_bytes)}
    except OSError as error:
        return {**base, "reason": "EXECUTION_ERROR", "elapsed_seconds": time.perf_counter() - start,
                "error": str(error)}
    elapsed = time.perf_counter() - start
    stdout_bytes, stderr_bytes = completed.stdout, completed.stderr
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    base.update({"elapsed_seconds": elapsed, "exit_code": completed.returncode,
                 "stdout": stdout, "stderr": stderr,
                 "stdout_sha256": _sha256(stdout_bytes),
                 "stderr_sha256": _sha256(stderr_bytes)})
    if completed.returncode != 0:
        return {**base, "reason": "PROCESS_ERROR"}
    try:
        payload = json.loads(stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError):
        return {**base, "reason": "INVALID_JSON"}
    if not isinstance(payload, dict) or payload.get("schema") != OBSERVATION_SCHEMA or payload.get("test_id") != test.test_id:
        return {**base, "reason": "OBSERVATION_BINDING_MISMATCH"}
    outcome = payload.get("outcome")
    calls = payload.get("subject_invocations")
    details = payload.get("details")
    if outcome not in {"PASS", "FAIL", "NON_EVALUABLE"} or not isinstance(calls, int) or isinstance(calls, bool) or calls < 0 or not isinstance(details, dict):
        return {**base, "reason": "INVALID_OBSERVATION_SCHEMA"}
    if outcome in {"PASS", "FAIL"} and calls == 0:
        return {**base, "reason": "NO_SUBJECT_EXECUTION"}
    try:
        json.dumps(details, allow_nan=False)
    except (TypeError, ValueError):
        return {**base, "reason": "INVALID_OBSERVATION_DETAILS"}
    if outcome == "NON_EVALUABLE":
        return {**base, "reason": "WORKER_NON_EVALUABLE", "subject_invocations": calls,
                "details": details}
    return {**base, "outcome": outcome, "subject_invocations": calls, "details": details}


def execute_plan(plan: TestPlan, contract: ComponentContract,
                 registry: Mapping[str, tuple[str, ...]], *, root: Path) -> dict[str, Any]:
    """Run all selected relations or abstain; never infer a pass from plan readiness."""

    verify_plan_contract(plan, contract)
    started = time.perf_counter()
    deadline = started + plan.wall_seconds
    records: list[dict[str, Any]] = []
    if plan.status == "READY":
        for test in plan.tests:
            records.append(_process_record(test, registry.get(test.test_id), deadline, root))
    complete = bool(records) and len(records) == len(plan.tests) and all(
        row["outcome"] in {"PASS", "FAIL"} for row in records
    )
    if plan.status != "READY":
        status, reason = "ABSTAIN", f"PLAN_{plan.reason_code or 'NOT_READY'}"
    elif any(row["outcome"] == "FAIL" for row in records):
        status, reason = "NONCONFORMANT", "OBSERVED_RELATION_FAILURE"
    elif not complete:
        status, reason = "ABSTAIN", "INCOMPLETE_EXECUTION"
    else:
        status, reason = "CONFORMANT", None
    return {
        "schema": REPORT_SCHEMA,
        "contract_id": plan.contract_id,
        "plan_digest": plan.digest,
        "plan_status": plan.status,
        "coverage_certificate": dict(plan.coverage_certificate),
        "status": status,
        "reason_code": reason,
        "selected_relations_executed": complete,
        "selected_relation_count": len(plan.tests),
        "reported_subject_invocations": sum(row["subject_invocations"] for row in records),
        "elapsed_seconds": time.perf_counter() - started,
        "observations": records,
        "evidence_scope": "selected_relations_only; worker_oracles_and_invocation_counts_are_not_independently_verified",
    }

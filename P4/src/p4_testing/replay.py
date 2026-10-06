"""Safe orchestration for unchanged historical buggy/fixed runners."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark import CapturedCase
from .models import CaseManifest, Outcome, PairResult, TestObservation, TestPlan
from .results import classify_pair


class ReplayError(RuntimeError):
    """Raised when replay metadata violates an immutable-runner requirement."""


@dataclass(frozen=True)
class ProcessRecord:
    status: str
    reason: str | None
    payload: Mapping[str, Any] | None
    stdout: str
    stderr: str
    exit_code: int | None
    elapsed_seconds: float
    command: tuple[str, ...]


@dataclass(frozen=True)
class CapturedPairEvaluation:
    complete_pair: bool
    runner_hash_matches: bool
    detected_by_at_least_one_strategy: bool
    pair_result: PairResult


def runner_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ReplayError(f"cannot read runner {path}: {error}") from error


def verify_runner_identity(left: Path, right: Path) -> str:
    left_hash = runner_sha256(left)
    right_hash = runner_sha256(right)
    if left_hash != right_hash:
        raise ReplayError(
            f"runner hash mismatch: {left_hash} != {right_hash}"
        )
    return left_hash


def run_json_process(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: Path | None = None,
) -> ProcessRecord:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("command must be a non-empty sequence of strings")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        return ProcessRecord(
            status="non_evaluable",
            reason="timeout",
            payload=None,
            stdout=(error.stdout or "") if isinstance(error.stdout, str) else "",
            stderr=(error.stderr or "") if isinstance(error.stderr, str) else "",
            exit_code=None,
            elapsed_seconds=time.perf_counter() - started,
            command=tuple(command),
        )
    except FileNotFoundError as error:
        return ProcessRecord(
            status="non_evaluable",
            reason="missing_executable",
            payload=None,
            stdout="",
            stderr=str(error),
            exit_code=None,
            elapsed_seconds=time.perf_counter() - started,
            command=tuple(command),
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return ProcessRecord(
            status="non_evaluable",
            reason="process_error",
            payload=None,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            elapsed_seconds=elapsed,
            command=tuple(command),
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ProcessRecord(
            status="non_evaluable",
            reason="invalid_json",
            payload=None,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            elapsed_seconds=elapsed,
            command=tuple(command),
        )
    if not isinstance(payload, Mapping):
        return ProcessRecord(
            status="non_evaluable",
            reason="invalid_json_shape",
            payload=None,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            elapsed_seconds=elapsed,
            command=tuple(command),
        )
    return ProcessRecord(
        status="ok",
        reason=None,
        payload=dict(payload),
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        elapsed_seconds=elapsed,
        command=tuple(command),
    )


def _observation(record: ProcessRecord, default_test_id: str) -> TestObservation:
    if record.status != "ok" or record.payload is None:
        return TestObservation(
            test_id=default_test_id,
            outcome=Outcome.NON_EVALUABLE,
            invocations=0,
            elapsed_seconds=record.elapsed_seconds,
            details={"reason": record.reason, "exit_code": record.exit_code},
        )
    payload = record.payload
    try:
        outcome = Outcome(str(payload["outcome"]))
        invocations = int(payload.get("invocations", 1))
    except (KeyError, TypeError, ValueError):
        return TestObservation(
            test_id=default_test_id,
            outcome=Outcome.REPRODUCTION_MISMATCH,
            invocations=0,
            elapsed_seconds=record.elapsed_seconds,
            details={"reason": "invalid_observation_payload"},
        )
    details = payload.get("details", {})
    if not isinstance(details, Mapping):
        details = {"reported_details": details}
    return TestObservation(
        test_id=str(payload.get("test_id", default_test_id)),
        outcome=outcome,
        invocations=invocations,
        elapsed_seconds=record.elapsed_seconds,
        details=dict(details),
    )


def _command(role: Mapping[str, Any]) -> tuple[str, ...]:
    raw = role.get("command")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ReplayError("version role requires a non-empty command")
    if not all(isinstance(item, str) and item for item in raw):
        raise ReplayError("version role command must contain only strings")
    return tuple(raw)


def _command_uses_runner(command: Sequence[str], runner: Path) -> bool:
    target = runner.resolve()
    for item in command:
        try:
            if Path(item).resolve() == target:
                return True
        except OSError:
            continue
    return False


def run_version_pair(
    case: CaseManifest,
    plan: TestPlan,
    timeout_seconds: float,
) -> PairResult:
    runner = Path(case.runner_path)
    digest = runner_sha256(runner)
    buggy_command = _command(case.buggy)
    fixed_command = _command(case.fixed)
    if not _command_uses_runner(buggy_command, runner) or not _command_uses_runner(fixed_command, runner):
        raise ReplayError("both version commands must use the declared unchanged runner")
    default_test_id = plan.tests[0].test_id if plan.tests else "replay.case"
    buggy_record = run_json_process(buggy_command, timeout_seconds)
    fixed_record = run_json_process(fixed_command, timeout_seconds)
    buggy_observation = _observation(buggy_record, default_test_id)
    fixed_observation = _observation(fixed_record, default_test_id)
    if buggy_observation.test_id != fixed_observation.test_id:
        buggy_observation = TestObservation(
            default_test_id,
            Outcome.REPRODUCTION_MISMATCH,
            buggy_observation.invocations,
            buggy_observation.elapsed_seconds,
            {"reason": "test_id_mismatch"},
        )
    decision = classify_pair(buggy_observation, fixed_observation)
    return PairResult(
        case_id=case.case_id,
        defect_family=case.defect_family,
        project=case.project,
        status=decision.status,
        detected=decision.detected,
        buggy=buggy_observation,
        fixed=fixed_observation,
        runner_sha256=digest,
        runner_hash_matches=True,
        plan_digest=plan.digest,
        metadata={
            "buggy_stdout_sha256": hashlib.sha256(buggy_record.stdout.encode("utf-8")).hexdigest(),
            "buggy_stderr_sha256": hashlib.sha256(buggy_record.stderr.encode("utf-8")).hexdigest(),
            "fixed_stdout_sha256": hashlib.sha256(fixed_record.stdout.encode("utf-8")).hexdigest(),
            "fixed_stderr_sha256": hashlib.sha256(fixed_record.stderr.encode("utf-8")).hexdigest(),
        },
    )


def evaluate_captured_pair(case: CapturedCase, plan: TestPlan) -> CapturedPairEvaluation:
    """Adapt already captured buggy/fixed evidence without rerunning or editing it."""

    try:
        capture_bytes = case.capture_path.read_bytes()
        capture = json.loads(capture_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read captured pair: {error}") from error
    if not isinstance(capture, Mapping) or capture.get("case_id") != case.case_id:
        raise ReplayError("captured pair case id mismatch")
    actual_hash = runner_sha256(case.runner_path)
    recorded_hash = str(capture.get("runner_sha256", "")).lower()
    runner_matches = (
        actual_hash == case.runner_sha256
        and recorded_hash == case.runner_sha256
        and capture.get("runner_unchanged_between_sides") is True
    )
    buggy_side = capture.get("buggy_side", {})
    fixed_side = capture.get("fixed_side", {})
    if not isinstance(buggy_side, Mapping) or not isinstance(fixed_side, Mapping):
        raise ReplayError("captured pair sides must be mappings")
    buggy_result = buggy_side.get("result", {})
    fixed_result = fixed_side.get("result", {})
    if not isinstance(buggy_result, Mapping) or not isinstance(fixed_result, Mapping):
        raise ReplayError("captured runner results must be mappings")
    buggy_confirmed = buggy_result.get("role_confirmed") is True
    fixed_confirmed = fixed_result.get("role_confirmed") is True
    complete = capture.get("complete_pair") is True and buggy_confirmed and fixed_confirmed
    test_id = plan.tests[0].test_id if plan.tests else "captured.reference"
    buggy = TestObservation(
        test_id,
        Outcome.FAIL if buggy_confirmed else Outcome.REPRODUCTION_MISMATCH,
        1,
        0.0,
        {"capture_role": "buggy", "capture_sha256": hashlib.sha256(capture_bytes).hexdigest()},
    )
    fixed = TestObservation(
        test_id,
        Outcome.PASS if fixed_confirmed else Outcome.REPRODUCTION_MISMATCH,
        1,
        0.0,
        {"capture_role": "fixed", "capture_sha256": hashlib.sha256(capture_bytes).hexdigest()},
    )
    decision = classify_pair(buggy, fixed)
    pair_result = PairResult(
        case_id=case.case_id,
        defect_family=case.defect_family,
        project=case.project,
        status=decision.status if runner_matches else "reproduction_mismatch",
        detected=decision.detected and runner_matches and complete,
        buggy=buggy,
        fixed=fixed,
        runner_sha256=actual_hash,
        runner_hash_matches=runner_matches,
        plan_digest=plan.digest,
        metadata={
            "capture_path": case.capture_path.name,
            "capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
            "detected_by": case.detected_by,
        },
    )
    return CapturedPairEvaluation(
        complete_pair=complete,
        runner_hash_matches=runner_matches,
        detected_by_at_least_one_strategy=bool(case.detected_by) and pair_result.detected,
        pair_result=pair_result,
    )

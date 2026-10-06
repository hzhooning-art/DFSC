"""Run one externally supplied baseline command under the preregistered budget."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


def command_digest(command: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(command), separators=(",", ":")).encode()).hexdigest()


def run_baseline_command(command: Sequence[str], timeout_seconds: float) -> dict[str, Any]:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("command must be a non-empty string sequence")
    started = time.perf_counter()
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout_seconds,
                                   check=False, shell=False)
    except subprocess.TimeoutExpired as error:
        return {"status": "abstain", "detected": False, "reason": "timeout",
                "elapsed_seconds": time.perf_counter() - started,
                "subject_invocations": 0, "command_sha256": command_digest(command),
                "stdout": error.stdout or "", "stderr": error.stderr or ""}
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return {"status": "abstain", "detected": False, "reason": "process_error",
                "elapsed_seconds": elapsed, "subject_invocations": 0,
                "command_sha256": command_digest(command), "stdout": completed.stdout,
                "stderr": completed.stderr, "exit_code": completed.returncode}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "abstain", "detected": False, "reason": "invalid_json",
                "elapsed_seconds": elapsed, "subject_invocations": 0,
                "command_sha256": command_digest(command), "stdout": completed.stdout,
                "stderr": completed.stderr}
    if not isinstance(payload, dict) or payload.get("status") not in {"detected", "missed"}:
        return {"status": "abstain", "detected": False, "reason": "invalid_result_schema",
                "elapsed_seconds": elapsed, "subject_invocations": 0,
                "command_sha256": command_digest(command), "stdout": completed.stdout,
                "stderr": completed.stderr}
    payload = dict(payload)
    payload.update({"elapsed_seconds": elapsed, "command_sha256": command_digest(command),
                    "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest()})
    return payload

"""Execute the unchanged SciPy #4130 runner in old and post-fix wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


TEST_ID = "replay.version_pair"
SCHEMA = "P4-Relation-Observation-v1"
RUNNER_SCHEMA = "P4-Role-Independent-Runner-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(outcome: str, calls: int, details: dict) -> dict:
    return {"schema": SCHEMA, "test_id": TEST_ID, "outcome": outcome,
            "subject_invocations": calls, "details": details}


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except (ValueError, AttributeError) as error:
        raise ValueError(f"unsupported numeric version: {value}") from error


def _role_run(python: Path, runner: Path, env: dict[str, str]) -> dict:
    command = [str(python), "-B", str(runner)]
    try:
        completed = subprocess.run(command, capture_output=True, check=False,
                                   timeout=8.0, env=env, shell=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "NOT_RUN", "error": str(error), "command": command}
    stdout, stderr = completed.stdout, completed.stderr
    record = {
        "status": "PROCESS_ERROR" if completed.returncode else "OBSERVED",
        "command": command,
        "exit_code": completed.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    if completed.returncode:
        return record
    try:
        payload = json.loads(stdout, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (ValueError, TypeError):
        record["status"] = "INVALID_JSON"
        return record
    if (not isinstance(payload, dict) or payload.get("schema") != RUNNER_SCHEMA or
            payload.get("case_id") != "scipy_4130" or payload.get("package") != "scipy" or
            payload.get("outcome") not in {"pass", "fail"} or
            not isinstance(payload.get("details"), dict)):
        record["status"] = "INVALID_OBSERVATION"
        return record
    record["payload"] = payload
    return record


def run(case: Path, contract: Path, runner: Path,
        legacy_python: Path, current_python: Path) -> dict:
    try:
        case_data = json.loads(case.read_text(encoding="utf-8"))
        contract_data = json.loads(contract.read_text(encoding="utf-8"))
        relation = contract_data["relations"]["version_pair"]
        if (case_data.get("case_id") != "scipy_4130" or
                contract_data.get("contract_id") != "scipy-fisher-exact-two-sided-v1" or
                relation.get("layer") != "replay" or
                relation.get("generator") != "unchanged_runner_pair" or
                relation.get("comparator") != "paired_role_semantics" or
                not {"historical_regression", "replay"}.issubset(relation.get("mechanisms", ()) )):
            raise ValueError("case or relation declaration mismatch")
        runner_hash = _sha256(runner)
        if runner_hash != case_data["runner_sha256"]:
            raise ValueError("frozen runner SHA-256 mismatch")
        if not legacy_python.is_file() or not current_python.is_file():
            raise ValueError("one or both Python interpreters are missing")
        fixed_release = _version_tuple(contract_data["provenance"]["fixed_release"])
        legacy_python_hash = _sha256(legacy_python)
        current_python_hash = _sha256(current_python)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation("NON_EVALUABLE", 0, {"reason": "INVALID_OR_MISSING_REPLAY_INPUT", "error": str(error)})

    legacy_env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "P4_SCIPY_SITE_PACKAGES", "P4_SCIPY_DLL_DIR"):
        legacy_env.pop(name, None)
    current_env = os.environ.copy()
    site = current_env.get("P4_SCIPY_SITE_PACKAGES")
    if site:
        fixture_dir = Path(__file__).resolve().parents[1] / "tests/reconstruction/fixtures"
        current_env["PYTHONPATH"] = os.pathsep.join((str(fixture_dir), site))

    buggy = _role_run(legacy_python, runner, legacy_env)
    fixed = _role_run(current_python, runner, current_env)
    details = {
        "runner_sha256": runner_hash,
        "legacy_python_sha256": legacy_python_hash,
        "current_python_sha256": current_python_hash,
        "buggy": buggy,
        "fixed": fixed,
        "same_runner_for_both_roles": True,
        "captured_pair_file_used_as_observation": False,
    }
    if buggy["status"] != "OBSERVED" or fixed["status"] != "OBSERVED":
        return _observation("NON_EVALUABLE", 0, {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    old, new = buggy["payload"], fixed["payload"]
    try:
        old_version = _version_tuple(old["version"])
        new_version = _version_tuple(new["version"])
        old_error = float(old["details"]["absolute_pvalue_error"])
        new_error = float(new["details"]["absolute_pvalue_error"])
        old_reference = float(old["details"]["reference_pvalue"])
        new_reference = float(new["details"]["reference_pvalue"])
        tolerance = float(contract_data["references"]["atol"])
        valid_numbers = all(math.isfinite(item) for item in
                            (old_error, new_error, old_reference, new_reference, tolerance))
        valid_roles = (old_version == (1, 8, 1) and new_version >= fixed_release and
                       old_version < new_version and old["outcome"] == "fail" and
                       old_error > tolerance and abs(old_reference - new_reference) <= tolerance)
    except (KeyError, TypeError, ValueError) as error:
        return _observation("NON_EVALUABLE", 0, {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    if not valid_numbers or not valid_roles:
        return _observation("NON_EVALUABLE", 2, {**details, "reason": "ROLE_REPRODUCTION_MISMATCH"})
    outcome = "PASS" if new["outcome"] == "pass" and new_error <= tolerance else "FAIL"
    return _observation(outcome, 2, {**details, "reason": "LIVE_UNCHANGED_RUNNER_PAIR",
                                     "buggy_version": old["version"], "fixed_version": new["version"],
                                     "reference_pvalue": new_reference,
                                     "buggy_absolute_error": old_error,
                                     "fixed_absolute_error": new_error,
                                     "atol": tolerance})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--legacy-python", required=True, type=Path)
    parser.add_argument("--current-python", required=True, type=Path)
    args = parser.parse_args()
    if args.test_id != TEST_ID:
        raise SystemExit(f"unsupported test_id: {args.test_id}")
    print(json.dumps(run(args.case, args.contract, args.runner,
                         args.legacy_python, args.current_python),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

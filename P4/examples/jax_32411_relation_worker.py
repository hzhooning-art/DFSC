"""Execute selected JAX #32411 property and historical-replay relations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


SCHEMA = "P4-Relation-Observation-v1"
RUNNER_SCHEMA = "P4-Role-Independent-Runner-v1"
CASE_ID = "jax_32411"
PROPERTY_ID = "property.gradient_accumulation_equivalence"
REPLAY_ID = "replay.version_pair"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(test_id: str, outcome: str, calls: int, details: dict) -> dict:
    return {"schema": SCHEMA, "test_id": test_id, "outcome": outcome,
            "subject_invocations": calls, "details": details}


def _version(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in value.split("."))
    if not parts:
        raise ValueError("empty version")
    return parts


def _inputs(test_id: str, contract_path: Path, case_path: Path,
            runner: Path, legacy: Path | None, current: Path) -> dict:
    source = contract_path.read_bytes()
    contract = json.loads(source)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if (contract.get("contract_id") != "jax-vjp-ref-gradient-accumulation-v1" or
            case.get("case_id") != CASE_ID or
            contract["domain"]["parameters"] !=
            {"layers": [2], "embedding_size": [32], "microbatches": [3], "microbatch_size": [4]} or
            contract["execution"]["dtypes"] != ["float32"] or
            contract["execution"]["devices"] != ["cpu"]):
        raise ValueError("unsupported case or contract domain")
    relation = contract["relations"][test_id.split(".", 1)[1]]
    expected = {
        PROPERTY_ID: ("property", "seeded_microbatch_problem", "max_absolute_gradient_error"),
        REPLAY_ID: ("replay", "unchanged_runner_pair", "paired_role_semantics"),
    }[test_id]
    if (relation.get("layer"), relation.get("generator"), relation.get("comparator")) != expected:
        raise ValueError("relation declaration mismatch")
    tolerance = contract["relations"]["gradient_accumulation_equivalence"]["parameters"]["atol"]
    if (not isinstance(tolerance, (float, int)) or isinstance(tolerance, bool) or
            not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("invalid gradient tolerance")
    fixed_release = _version(contract["provenance"]["fixed_release"])
    if runner.resolve() != (Path(__file__).resolve().parents[1] / case["runner_path"]).resolve():
        raise ValueError("frozen runner path mismatch")
    runner_hash = _sha256(runner)
    if runner_hash != case["runner_sha256"]:
        raise ValueError("frozen runner SHA-256 mismatch")
    if not current.is_file() or (test_id == REPLAY_ID and (legacy is None or not legacy.is_file())):
        raise ValueError("one or both required Python interpreters are missing")
    return {"contract_sha256": hashlib.sha256(source).hexdigest(),
            "runner_sha256": runner_hash, "current_python_sha256": _sha256(current),
            "legacy_python_sha256": _sha256(legacy) if test_id == REPLAY_ID else None,
            "atol": float(tolerance), "fixed_release": fixed_release}


def _role_run(python: Path, runner: Path) -> dict:
    argv = [str(python), "-B", str(runner)]
    child_env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "P4_SCIPY_SITE_PACKAGES", "P4_SCIPY_DLL_DIR",
                 "KMP_DUPLICATE_LIB_OK"):
        child_env.pop(name, None)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(argv, capture_output=True, check=False,
                                timeout=12.0, env=child_env, shell=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "NOT_RUN", "command": argv, "error": str(error)}
    stdout, stderr = result.stdout, result.stderr
    record = {"status": "PROCESS_ERROR" if result.returncode else "OBSERVED",
              "command": argv, "exit_code": result.returncode,
              "stdout": stdout.decode("utf-8", errors="replace"),
              "stderr": stderr.decode("utf-8", errors="replace"),
              "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
              "stderr_sha256": hashlib.sha256(stderr).hexdigest()}
    if result.returncode:
        return record
    try:
        payload = json.loads(stdout, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError):
        record["status"] = "INVALID_JSON"
        return record
    if (not isinstance(payload, dict) or payload.get("schema") != RUNNER_SCHEMA or
            payload.get("case_id") != CASE_ID or payload.get("package") != "jax" or
            payload.get("outcome") not in {"pass", "fail"} or
            payload.get("invocations") != 2 or not isinstance(payload.get("details"), dict)):
        record["status"] = "INVALID_OBSERVATION"
        return record
    record["payload"] = payload
    return record


def _metrics(payload: dict) -> tuple[tuple[int, ...], float, float]:
    details = payload["details"]
    if (details["reference_method"] != "value-and-grad-scan-accumulation" or
            details["shape"] != [2, 32, 32] or details["microbatches"] != 3):
        raise ValueError("runner semantics differ from frozen case")
    gradient_error = float(details["max_absolute_gradient_error"])
    value_error = float(details["max_absolute_value_error"])
    if (not math.isfinite(gradient_error) or not math.isfinite(value_error) or
            gradient_error < 0 or value_error < 0):
        raise ValueError("invalid numerical error")
    return _version(payload["version"]), gradient_error, value_error


def run(test_id: str, contract_path: Path, case_path: Path, runner: Path,
        current: Path, legacy: Path | None = None) -> dict:
    if test_id not in {PROPERTY_ID, REPLAY_ID}:
        raise ValueError("unsupported test_id")
    try:
        binding = _inputs(test_id, contract_path, case_path, runner, legacy, current)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {"reason": "INVALID_OR_MISSING_EXECUTION_INPUT", "error": str(error)})
    if test_id == PROPERTY_ID:
        fixed = _role_run(current, runner)
        details = {**binding, "fixed": fixed, "historical_pair_executed": False,
                   "captured_pair_file_used_as_observation": False}
        details["fixed_release"] = list(binding["fixed_release"])
        if fixed["status"] != "OBSERVED":
            return _observation(test_id, "NON_EVALUABLE", 0,
                                {**details, "reason": "CURRENT_EXECUTION_INCOMPLETE"})
        try:
            version, gradient_error, value_error = _metrics(fixed["payload"])
        except (KeyError, TypeError, ValueError) as error:
            return _observation(test_id, "NON_EVALUABLE", 2,
                                {**details, "reason": "INVALID_CURRENT_SEMANTICS", "error": str(error)})
        if version < binding["fixed_release"]:
            return _observation(test_id, "NON_EVALUABLE", 2,
                                {**details, "reason": "CURRENT_VERSION_PRECEDES_FIX"})
        passed = (fixed["payload"]["outcome"] == "pass" and
                  gradient_error < binding["atol"] and value_error < binding["atol"])
        return _observation(test_id, "PASS" if passed else "FAIL", 2,
                            {**details, "reason": "LIVE_CURRENT_GRADIENT_COMPARISON",
                             "current_version": fixed["payload"]["version"],
                             "max_absolute_gradient_error": gradient_error,
                             "max_absolute_value_error": value_error})
    buggy = _role_run(legacy, runner)
    fixed = _role_run(current, runner)
    details = {**binding, "buggy": buggy, "fixed": fixed,
               "same_runner_for_both_roles": True,
               "captured_pair_file_used_as_observation": False}
    details["fixed_release"] = list(binding["fixed_release"])
    if buggy["status"] != "OBSERVED" or fixed["status"] != "OBSERVED":
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    try:
        old_version, old_gradient, old_value = _metrics(buggy["payload"])
        new_version, new_gradient, new_value = _metrics(fixed["payload"])
        valid_old = (old_version == (0, 7, 2) and buggy["payload"]["outcome"] == "fail" and
                     old_gradient >= binding["atol"] and old_value < binding["atol"])
        valid_new_version = new_version >= binding["fixed_release"] and new_version > old_version
    except (KeyError, TypeError, ValueError) as error:
        return _observation(test_id, "NON_EVALUABLE", 4,
                            {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    if not valid_old or not valid_new_version:
        return _observation(test_id, "NON_EVALUABLE", 4,
                            {**details, "reason": "HISTORICAL_ROLE_MISMATCH"})
    passed = (fixed["payload"]["outcome"] == "pass" and
              new_gradient < binding["atol"] and new_value < binding["atol"])
    return _observation(test_id, "PASS" if passed else "FAIL", 4,
                        {**details, "reason": "LIVE_UNCHANGED_RUNNER_PAIR",
                         "buggy_version": buggy["payload"]["version"],
                         "fixed_version": fixed["payload"]["version"],
                         "buggy_gradient_error": old_gradient,
                         "fixed_gradient_error": new_gradient})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--legacy-python", type=Path)
    parser.add_argument("--current-python", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.test_id, args.contract, args.case, args.runner,
                 args.current_python, args.legacy_python)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

"""Run the three selected relations for the frozen PyTorch #80770 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "pytorch_80770"
RUNNER_SCHEMA = "P4-Historical-Pair-Runner-v1"
OBSERVATION_SCHEMA = "P4-Relation-Observation-v1"
OLD = ROOT / ".runtime/p4_torch111_replay/python310/python.exe"
NEW = ROOT.parent / "P1/.venv/Scripts/python.exe"
TESTS = {
    "reference.analytic_derivative": ("reference", "positive_y_at_zero_x", "autograd_matches_log_y"),
    "execution.repeatability": ("execution", "repeat_identical_request", "exact_repeat"),
    "replay.version_pair": ("replay", "unchanged_runner_pair", "paired_role_semantics"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(test_id: str, outcome: str, calls: int, details: dict) -> dict:
    return {"schema": OBSERVATION_SCHEMA, "test_id": test_id,
            "outcome": outcome, "subject_invocations": calls, "details": details}


def role_run(python: Path, runner: Path, role: str) -> dict:
    argv = [str(python), "-B", str(runner), "--case", CASE_ID, "--expected-role", role]
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "P4_SCIPY_SITE_PACKAGES", "P4_SCIPY_DLL_DIR",
                "KMP_DUPLICATE_LIB_OK"):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(argv, shell=False, check=False, capture_output=True,
                                timeout=18, env=environment)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "NOT_RUN", "command": argv, "error": str(error)}
    record = {"status": "PROCESS_ERROR" if result.returncode else "OBSERVED",
              "command": argv, "exit_code": result.returncode,
              "stdout": result.stdout.decode("utf-8", errors="replace"),
              "stderr": result.stderr.decode("utf-8", errors="replace"),
              "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
              "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}
    if result.returncode:
        return record
    try:
        payload = json.loads(result.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (TypeError, ValueError):
        record["status"] = "INVALID_JSON"
        return record
    if (not isinstance(payload, dict) or payload.get("schema") != RUNNER_SCHEMA or
            payload.get("case_id") != CASE_ID or payload.get("expected_role") != role or
            not isinstance(payload.get("observation"), dict)):
        record["status"] = "INVALID_OBSERVATION"
        return record
    record["payload"] = payload
    return record


def metrics(record: dict, role: str, tolerance: float) -> tuple[str, float]:
    payload = record["payload"]
    result = payload["observation"]
    version = payload["torch"]
    observed = float(result["observed_gradient"])
    expected = math.log(2.0)
    error = float(result["maximum_absolute_error"])
    if (not isinstance(version, str) or not math.isfinite(observed) or
            not math.isfinite(error) or error < 0 or
            result.get("executed") is not True or result.get("device") != "cpu" or
            float(result["expected_gradient"]) != expected or
            not math.isclose(error, abs(observed - expected), abs_tol=1e-15) or
            result.get("matches_fixed_expectation") is not (error <= tolerance) or
            result.get("matches_reported_bug") is not (abs(observed) <= 1e-15)):
        raise ValueError("frozen runner numerical or role semantics mismatch")
    if role == "buggy" and (version != "1.11.0+cpu" or abs(observed) > 1e-15 or
                            payload.get("role_confirmed") is not True):
        raise ValueError("historical buggy wheel was not reproduced")
    if role == "fixed" and not version.startswith("2.11.0+"):
        raise ValueError("unexpected current wheel version")
    return version, observed


def run(test_id: str) -> dict:
    if test_id not in TESTS:
        raise ValueError("unsupported relation")
    contract_path = ROOT / "contracts/cases/pytorch_80770.json"
    case_path = ROOT / "benchmark/cases/pytorch_80770.json"
    try:
        contract_bytes = contract_path.read_bytes()
        contract = json.loads(contract_bytes)
        case = json.loads(case_path.read_text(encoding="utf-8"))
        runner = ROOT / case["runner_path"]
        relation = contract["relations"][test_id.split(".", 1)[1]]
        if (contract.get("contract_id") != "pytorch-xlogy-derivative-v1" or
                case.get("case_id") != CASE_ID or
                (relation["layer"], relation["generator"], relation["comparator"]) != TESTS[test_id] or
                contract["domain"]["parameters"] != {"x": [0.0], "y": [2.0]} or
                contract["execution"]["dtypes"] != ["float64"] or
                contract["execution"]["devices"] != ["cpu"] or
                runner.resolve() != (ROOT / "experiments/pytorch_historical_pair_runner.py").resolve() or
                digest(runner) != case["runner_sha256"] or
                not NEW.is_file() or (test_id == "replay.version_pair" and not OLD.is_file())):
            raise ValueError("contract, frozen runner or local interpreter mismatch")
        tolerance = float(contract["relations"]["analytic_derivative"]["parameters"]["atol"])
        if not math.isfinite(tolerance) or tolerance != 1e-12:
            raise ValueError("unsupported analytic tolerance")
        binding = {"contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                   "runner_sha256": digest(runner), "current_python_sha256": digest(NEW),
                   "legacy_python_sha256": digest(OLD) if test_id == "replay.version_pair" else None,
                   "analytic_expected_gradient": math.log(2.0), "atol": tolerance,
                   "captured_pair_file_used_as_observation": False}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return observation(test_id, "NON_EVALUABLE", 0,
                           {"reason": "INVALID_OR_MISSING_EXECUTION_INPUT", "error": str(error)})

    if test_id == "replay.version_pair":
        records = {"buggy": role_run(OLD, runner, "buggy"),
                   "fixed": role_run(NEW, runner, "fixed")}
    elif test_id == "execution.repeatability":
        records = {"fixed_first": role_run(NEW, runner, "fixed"),
                   "fixed_second": role_run(NEW, runner, "fixed")}
    else:
        records = {"fixed": role_run(NEW, runner, "fixed")}
    details = {**binding, **records}
    if any(record["status"] != "OBSERVED" for record in records.values()):
        return observation(test_id, "NON_EVALUABLE", 0,
                           {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    try:
        values = {name: metrics(record, "buggy" if name == "buggy" else "fixed", tolerance)
                  for name, record in records.items()}
    except (KeyError, TypeError, ValueError) as error:
        return observation(test_id, "NON_EVALUABLE", len(records),
                           {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    if test_id == "execution.repeatability":
        passed = values["fixed_first"] == values["fixed_second"]
        reason = "TWO_INDEPENDENT_FIXED_PROCESSES_EXACT_NUMERIC_REPEAT"
    elif test_id == "replay.version_pair":
        passed = values["buggy"][1] == 0.0 and abs(values["fixed"][1] - math.log(2.0)) <= tolerance
        reason = "LIVE_UNCHANGED_RUNNER_PAIR"
    else:
        passed = abs(values["fixed"][1] - math.log(2.0)) <= tolerance
        reason = "LIVE_ANALYTIC_DERIVATIVE_CHECK"
    return observation(test_id, "PASS" if passed else "FAIL", len(records),
                       {**details, "reason": reason, "versions_and_gradients": values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    print(json.dumps(run(parser.parse_args().test_id), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

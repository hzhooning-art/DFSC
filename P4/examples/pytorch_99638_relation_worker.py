"""Execute selected PyTorch #99638 relations with isolated wheel interpreters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".runtime/p4_torch1131_replay/python310/python.exe"
NEW = ROOT.parent / "P1/.venv/Scripts/python.exe"
CASE_ID = "pytorch_99638"
SCHEMA = "P4-Relation-Observation-v1"
RUNNER_SCHEMA = "P4-Role-Independent-Runner-v1"
TESTS = {
    "property.selection_gradient_equivalence":
        ("property", "equivalent_masked_and_index_selection", "bias_gradients_allclose"),
    "execution.deterministic_repeat":
        ("execution", "repeat_seeded_graph", "exact_repeat"),
    "replay.version_pair":
        ("replay", "unchanged_runner_pair", "paired_role_semantics"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(test_id: str, outcome: str, calls: int, details: dict) -> dict:
    return {"schema": SCHEMA, "test_id": test_id, "outcome": outcome,
            "subject_invocations": calls, "details": details}


def role_run(python: Path, runner: Path) -> dict:
    argv = [str(python), "-B", str(runner)]
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "P4_SCIPY_SITE_PACKAGES", "P4_SCIPY_DLL_DIR",
                "KMP_DUPLICATE_LIB_OK"):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(argv, check=False, shell=False, capture_output=True,
                                timeout=18.0, env=environment)
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
        payload = json.loads(result.stdout,
                             parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError):
        record["status"] = "INVALID_JSON"
        return record
    if (not isinstance(payload, dict) or payload.get("schema") != RUNNER_SCHEMA or
            payload.get("case_id") != CASE_ID or payload.get("package") != "torch" or
            payload.get("invocations") != 2 or not isinstance(payload.get("details"), dict)):
        record["status"] = "INVALID_OBSERVATION"
        return record
    record["payload"] = payload
    return record


def checked_metrics(record: dict, tolerance: float) -> tuple[str, float, tuple[float, ...], tuple[float, ...]]:
    payload = record["payload"]
    details = payload["details"]
    observed, reference = details["observed"], details["reference"]
    if (not isinstance(observed, list) or not isinstance(reference, list) or
            len(observed) != 5 or len(reference) != 5):
        raise ValueError("unexpected gradient vector shape")
    observed = tuple(float(value) for value in observed)
    reference = tuple(float(value) for value in reference)
    if not all(math.isfinite(value) for value in (*observed, *reference)):
        raise ValueError("nonfinite gradient vector")
    error = float(details["max_absolute_gradient_error"])
    actual = max(abs(left - right) for left, right in zip(observed, reference))
    if (not math.isfinite(error) or error < 0 or
            not math.isclose(error, actual, rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError("reported error disagrees with vector entries")
    allclose = all(abs(left - right) <= tolerance + tolerance * abs(right)
                   for left, right in zip(observed, reference))
    if payload.get("outcome") != ("pass" if allclose else "fail"):
        raise ValueError("runner outcome disagrees with vector comparison")
    version = payload["version"]
    if version not in ("1.13.1+cpu", "2.11.0+cu128"):
        raise ValueError("unexpected PyTorch version")
    return version, error, observed, reference


def run(test_id: str) -> dict:
    if test_id not in TESTS:
        raise ValueError("unsupported relation")
    try:
        contract_path = ROOT / "contracts/cases/pytorch_99638.json"
        case_path = ROOT / "benchmark/cases/pytorch_99638.json"
        contract_bytes = contract_path.read_bytes()
        contract = json.loads(contract_bytes)
        case = json.loads(case_path.read_text(encoding="utf-8"))
        runner = ROOT / case["runner_path"]
        relation = contract["relations"][test_id.split(".", 1)[1]]
        tolerance = float(contract["relations"]["selection_gradient_equivalence"]
                          ["parameters"]["atol"])
        if (contract.get("contract_id") != "pytorch-masked-select-gradient-v1" or
                case.get("case_id") != CASE_ID or
                (relation["layer"], relation["generator"], relation["comparator"]) != TESTS[test_id] or
                contract["domain"]["parameters"] != {"batch": [3], "time": [4], "features": [5]} or
                contract["execution"]["dtypes"] != ["float32"] or
                contract["execution"]["devices"] != ["cpu"] or
                contract["execution"]["seed"] != 42 or
                not math.isfinite(tolerance) or tolerance != 1e-10 or
                float(contract["relations"]["selection_gradient_equivalence"]
                      ["parameters"]["rtol"]) != tolerance or
                runner.resolve() != (ROOT / "benchmark/runners/pytorch/pytorch_99638.py").resolve() or
                digest(runner) != case["runner_sha256"] or
                not NEW.is_file() or (test_id == "replay.version_pair" and not OLD.is_file())):
            raise ValueError("contract, frozen runner or interpreter mismatch")
        binding = {"contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                   "runner_sha256": digest(runner), "current_python_sha256": digest(NEW),
                   "legacy_python_sha256": digest(OLD) if test_id == "replay.version_pair" else None,
                   "atol": tolerance, "rtol": tolerance,
                   "captured_pair_file_used_as_observation": False}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return observation(test_id, "NON_EVALUABLE", 0,
                           {"reason": "INVALID_OR_MISSING_EXECUTION_INPUT", "error": str(error)})
    if test_id == "replay.version_pair":
        records = {"buggy": role_run(OLD, runner), "fixed": role_run(NEW, runner)}
    elif test_id == "execution.deterministic_repeat":
        records = {"fixed_first": role_run(NEW, runner),
                   "fixed_second": role_run(NEW, runner)}
    else:
        records = {"fixed": role_run(NEW, runner)}
    details = {**binding, **records}
    if any(record["status"] != "OBSERVED" for record in records.values()):
        return observation(test_id, "NON_EVALUABLE", 0,
                           {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    calls = 2 * len(records)
    try:
        values = {name: checked_metrics(record, tolerance) for name, record in records.items()}
    except (KeyError, TypeError, ValueError) as error:
        return observation(test_id, "NON_EVALUABLE", calls,
                           {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    if ("buggy" in values and
            (values["buggy"][0] != "1.13.1+cpu" or values["buggy"][1] <= tolerance or
             records["buggy"]["payload"]["outcome"] != "fail")):
        return observation(test_id, "NON_EVALUABLE", calls,
                           {**details, "reason": "HISTORICAL_ROLE_MISMATCH"})
    if any(value[0] != "2.11.0+cu128" for name, value in values.items() if name != "buggy"):
        return observation(test_id, "NON_EVALUABLE", calls,
                           {**details, "reason": "CURRENT_VERSION_MISMATCH"})
    if test_id == "execution.deterministic_repeat":
        passed = values["fixed_first"] == values["fixed_second"]
        reason = "TWO_INDEPENDENT_SEEDED_CURRENT_PROCESSES_EXACT_REPEAT"
    else:
        passed = values["fixed"][1] <= tolerance and records["fixed"]["payload"]["outcome"] == "pass"
        reason = ("LIVE_UNCHANGED_RUNNER_PAIR" if test_id == "replay.version_pair" else
                  "LIVE_MASKED_VS_INDEXED_BIAS_GRADIENT_COMPARISON")
    return observation(test_id, "PASS" if passed else "FAIL", calls,
                       {**details, "reason": reason,
                        "versions_and_max_errors": {name: list(value[:2]) for name, value in values.items()}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    print(json.dumps(run(parser.parse_args().test_id), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

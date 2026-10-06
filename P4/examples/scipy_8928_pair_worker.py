"""Replay the frozen SciPy #8928 runner in historical and current wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


TEST_ID = "replay.version_pair"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(outcome: str, calls: int, details: dict) -> dict:
    return {"schema": "P4-Relation-Observation-v1", "test_id": TEST_ID,
            "outcome": outcome, "subject_invocations": calls, "details": details}


def _version(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"invalid numeric version: {value}") from error


def _role_run(python: Path, runner: Path, env: dict[str, str]) -> dict:
    command = [str(python), "-B", str(runner)]
    try:
        result = subprocess.run(command, capture_output=True, check=False,
                                timeout=8.0, env=env, shell=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "NOT_RUN", "command": command, "error": str(error)}
    stdout, stderr = result.stdout, result.stderr
    row = {"status": "PROCESS_ERROR" if result.returncode else "OBSERVED",
           "command": command, "exit_code": result.returncode,
           "stdout": stdout.decode("utf-8", errors="replace"),
           "stderr": stderr.decode("utf-8", errors="replace"),
           "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
           "stderr_sha256": hashlib.sha256(stderr).hexdigest()}
    if result.returncode:
        return row
    try:
        payload = json.loads(stdout, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (TypeError, ValueError):
        row["status"] = "INVALID_JSON"
        return row
    if (not isinstance(payload, dict) or payload.get("schema") != "P4-Role-Independent-Runner-v1" or
            payload.get("case_id") != "scipy_8928" or payload.get("package") != "scipy" or
            payload.get("outcome") not in {"pass", "fail"} or
            payload.get("invocations") != 2 or not isinstance(payload.get("details"), dict)):
        row["status"] = "INVALID_OBSERVATION"
        return row
    row["payload"] = payload
    return row


def run(case: Path, contract: Path, runner: Path,
        legacy_python: Path, current_python: Path) -> dict:
    try:
        case_data = json.loads(case.read_text(encoding="utf-8"))
        contract_data = json.loads(contract.read_text(encoding="utf-8"))
        relation = contract_data["relations"]["version_pair"]
        if (case_data.get("case_id") != "scipy_8928" or
                contract_data.get("contract_id") != "scipy-normal-expect-location-v1" or
                relation.get("layer") != "replay" or
                relation.get("generator") != "unchanged_runner_pair" or
                relation.get("comparator") != "paired_role_semantics" or
                not {"historical_regression", "replay"}.issubset(relation.get("mechanisms", ()))):
            raise ValueError("case or relation declaration mismatch")
        if runner.resolve() != (Path(__file__).resolve().parents[1] / case_data["runner_path"]).resolve():
            raise ValueError("runner path differs from frozen case manifest")
        runner_hash = _sha256(runner)
        if runner_hash != case_data["runner_sha256"]:
            raise ValueError("frozen runner SHA-256 mismatch")
        if not legacy_python.is_file() or not current_python.is_file():
            raise ValueError("one or both Python interpreters are missing")
        fixed_release = _version(contract_data["provenance"]["fixed_release"])
        tolerance = float(contract_data["relations"]["analytic_mean"]["parameters"]["atol"])
        locations = contract_data["domain"]["parameters"]["loc"]
        scales = contract_data["domain"]["parameters"]["scale"]
        if (not math.isfinite(tolerance) or tolerance <= 0 or
                locations != [36.0, 10.0] or scales != [1.0, 0.1]):
            raise ValueError("unsupported replay cases or tolerance")
        legacy_hash, current_hash = _sha256(legacy_python), _sha256(current_python)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation("NON_EVALUABLE", 0,
                            {"reason": "INVALID_OR_MISSING_REPLAY_INPUT", "error": str(error)})

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
    details = {"runner_sha256": runner_hash, "legacy_python_sha256": legacy_hash,
               "current_python_sha256": current_hash, "buggy": buggy, "fixed": fixed,
               "same_runner_for_both_roles": True,
               "captured_pair_file_used_as_observation": False}
    if buggy["status"] != "OBSERVED" or fixed["status"] != "OBSERVED":
        return _observation("NON_EVALUABLE", 0,
                            {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    old, new = buggy["payload"], fixed["payload"]
    try:
        old_version, new_version = _version(old["version"]), _version(new["version"])
        checks_old, checks_new = old["details"]["checks"], new["details"]["checks"]
        if len(checks_old) != 2 or len(checks_new) != 2:
            raise ValueError("expected two checks per role")
        for checks in (checks_old, checks_new):
            for item, loc, scale in zip(checks, locations, scales):
                if (float(item["loc"]) != loc or float(item["scale"]) != scale or
                        float(item["analytic_reference"]) != loc or
                        not math.isfinite(float(item["observed"])) or
                        not math.isfinite(float(item["absolute_error"])) or
                        not math.isclose(float(item["absolute_error"]),
                                         abs(float(item["observed"]) - loc),
                                         rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError("invalid check semantics")
        old_error = max(float(item["absolute_error"]) for item in checks_old)
        new_error = max(float(item["absolute_error"]) for item in checks_new)
        if (old_version != (1, 8, 1) or new_version < fixed_release or
                old_version >= new_version or old["outcome"] != "fail" or
                old_error <= tolerance):
            raise ValueError("buggy/fixed role mismatch")
    except (KeyError, TypeError, ValueError) as error:
        return _observation("NON_EVALUABLE", 4,
                            {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    outcome = "PASS" if new["outcome"] == "pass" and new_error <= tolerance else "FAIL"
    return _observation(outcome, 4,
                        {**details, "reason": "LIVE_UNCHANGED_RUNNER_PAIR",
                         "buggy_version": old["version"], "fixed_version": new["version"],
                         "buggy_max_absolute_error": old_error,
                         "fixed_max_absolute_error": new_error, "atol": tolerance})


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

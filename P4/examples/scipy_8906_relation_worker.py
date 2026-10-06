"""Execute the three declared SciPy #8906 relations against installed wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


SCHEMA = "P4-Relation-Observation-v1"
CASE_ID = "scipy_8906"
EXPECTED = [[0.5, 1.0, 1.5]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(test_id: str, outcome: str, calls: int, details: dict) -> dict:
    return {"schema": SCHEMA, "test_id": test_id, "outcome": outcome,
            "subject_invocations": calls, "details": details}


def _version(raw: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in raw.split("."))
    if not parts:
        raise ValueError("empty version")
    return parts


def _contract(path: Path, test_id: str) -> tuple[dict, dict, str]:
    source = path.read_bytes()
    contract = json.loads(source)
    if contract.get("contract_id") != "scipy-solve-banded-representation-v1":
        raise ValueError("wrong contract_id")
    relation = contract["relations"][test_id.split(".", 1)[1]]
    expected = {
        "reference.reference_solution": ("reference", "singleton_diagonal_system", "allclose_to_analytic_solution"),
        "property.representation_equivalence": ("property", "equivalent_band_representations", "allclose_between_representations"),
        "replay.version_pair": ("replay", "unchanged_runner_pair", "paired_role_semantics"),
    }[test_id]
    if (relation.get("layer"), relation.get("generator"), relation.get("comparator")) != expected:
        raise ValueError("relation declaration mismatch")
    if (contract["domain"]["parameters"]["lower_bands"] != [0, 1] or
            contract["domain"]["parameters"]["upper_bands"] != [0, 1]):
        raise ValueError("unsupported band domain")
    if test_id != "replay.version_pair":
        params = relation["parameters"]
        for key in ("atol", "rtol"):
            value = params[key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid {key}")
    return contract, relation, hashlib.sha256(source).hexdigest()


def _current_paths() -> None:
    site = os.environ.get("P4_SCIPY_SITE_PACKAGES")
    dll = os.environ.get("P4_SCIPY_DLL_DIR")
    if site:
        sys.path.insert(0, site)
    if dll and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll)


def _close(actual: list, expected: list, rtol: float, atol: float) -> bool:
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            return False
        for a, b in zip(actual_row, expected_row):
            if not isinstance(a, (int, float)) or not math.isfinite(a) or not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                return False
    return True


def _current_relation(test_id: str, relation: dict, contract_hash: str) -> dict:
    try:
        _current_paths()
        import scipy
        import numpy as np
        from scipy.linalg import solve_banded
    except (ImportError, OSError, ValueError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0, {"reason": "SCIPY_UNAVAILABLE", "error": str(error)})
    rhs = np.asarray([[1.0, 2.0, 3.0]])
    cases = [((0, 0), np.asarray([[2.0]]), "conventional")]
    if test_id == "property.representation_equivalence":
        cases.append(((1, 1), np.asarray([[0.0], [2.0], [0.0]]), "padded"))
    observed = {}
    for bands, matrix, label in cases:
        try:
            solution = np.asarray(solve_banded(bands, matrix, rhs.copy())).tolist()
        except Exception as error:
            return _observation(test_id, "FAIL", len(observed) + 1,
                                {"reason": "SUBJECT_EXCEPTION", "role": label,
                                 "error": type(error).__name__, "message": str(error),
                                 "prior_observations": observed, "scipy_version": scipy.__version__})
        observed[label] = solution
    rtol, atol = relation["parameters"]["rtol"], relation["parameters"]["atol"]
    if test_id == "reference.reference_solution":
        passed = _close(observed["conventional"], EXPECTED, rtol, atol)
    else:
        # Equivalence alone is insufficient: two equally wrong outputs must fail.
        passed = (_close(observed["conventional"], observed["padded"], rtol, atol) and
                  all(_close(value, EXPECTED, rtol, atol) for value in observed.values()))
    return _observation(test_id, "PASS" if passed else "FAIL", len(cases),
                        {"observed": observed, "analytic_solution": EXPECTED,
                         "rtol": rtol, "atol": atol, "scipy_version": scipy.__version__,
                         "scipy_file": str(Path(scipy.__file__).resolve()),
                         "python_executable": sys.executable,
                         "contract_sha256": contract_hash, "historical_pair_executed": False})


def _run_role(python: Path, runner: Path, role: str, env: dict[str, str]) -> dict:
    argv = [str(python), "-B", str(runner), "--expected-role", role]
    try:
        result = subprocess.run(argv, capture_output=True, check=False, timeout=8.0,
                                env=env, shell=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "NOT_RUN", "command": argv, "error": str(error)}
    stdout, stderr = result.stdout, result.stderr
    record = {"status": "PROCESS_ERROR" if result.returncode else "OBSERVED", "command": argv,
              "exit_code": result.returncode, "stdout": stdout.decode("utf-8", errors="replace"),
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
    if (not isinstance(payload, dict) or payload.get("schema") != "P4-SciPy-Historical-Pair-Runner-v1" or
            payload.get("case_id") != CASE_ID or payload.get("expected_role") != role or
            not isinstance(payload.get("observation"), dict)):
        record["status"] = "INVALID_OBSERVATION"
        return record
    record["payload"] = payload
    return record


def _pair_relation(test_id: str, case_path: Path, runner: Path,
                   legacy: Path, current: Path, contract_hash: str) -> dict:
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if case.get("case_id") != CASE_ID or runner.resolve() != (Path(__file__).resolve().parents[1] / case["runner_path"]).resolve():
            raise ValueError("frozen case/runner path mismatch")
        runner_hash = _sha256(runner)
        if runner_hash != case["runner_sha256"]:
            raise ValueError("frozen runner SHA-256 mismatch")
        if not legacy.is_file() or not current.is_file():
            raise ValueError("one or both interpreters are missing")
        hashes = {"runner_sha256": runner_hash, "legacy_python_sha256": _sha256(legacy),
                  "current_python_sha256": _sha256(current), "contract_sha256": contract_hash}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {"reason": "INVALID_OR_MISSING_REPLAY_INPUT", "error": str(error)})
    old_env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "P4_SCIPY_SITE_PACKAGES", "P4_SCIPY_DLL_DIR"):
        old_env.pop(name, None)
    new_env = os.environ.copy()
    site = new_env.get("P4_SCIPY_SITE_PACKAGES")
    if site:
        fixture = Path(__file__).resolve().parents[1] / "tests/reconstruction/fixtures"
        new_env["PYTHONPATH"] = os.pathsep.join((str(fixture), site))
    old = _run_role(legacy, runner, "buggy", old_env)
    new = _run_role(current, runner, "fixed", new_env)
    details = {**hashes, "buggy": old, "fixed": new, "same_runner_for_both_roles": True,
               "captured_pair_file_used_as_observation": False}
    if old["status"] != "OBSERVED" or new["status"] != "OBSERVED":
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    try:
        buggy, fixed = old["payload"], new["payload"]
        a, b = _version(buggy["scipy"]), _version(fixed["scipy"])
        old_obs, new_obs = buggy["observation"], fixed["observation"]
        valid_old = (a == (1, 14, 1) and buggy["role_confirmed"] is True and
                     old_obs["matches_reported_bug"] is True and
                     old_obs["conventional_u0"]["executed"] is False and
                     old_obs["conventional_u0"]["exception"] == "IndexError" and
                     old_obs["padded_u1"]["executed"] is True and
                     _close(old_obs["padded_u1"]["solution"], EXPECTED, 1e-12, 1e-12))
        valid_new = (b >= (1, 15, 0) and a < b and fixed["role_confirmed"] is True and
                     new_obs["matches_fixed_expectation"] is True and
                     all(new_obs[key]["executed"] is True and
                         _close(new_obs[key]["solution"], EXPECTED, 1e-12, 1e-12)
                         for key in ("conventional_u0", "padded_u1")))
    except (KeyError, TypeError, ValueError) as error:
        return _observation(test_id, "NON_EVALUABLE", 4,
                            {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    if not valid_old:
        return _observation(test_id, "NON_EVALUABLE", 4,
                            {**details, "reason": "BUGGY_ROLE_NOT_REPRODUCED"})
    return _observation(test_id, "PASS" if valid_new else "FAIL", 4,
                        {**details, "reason": "LIVE_UNCHANGED_RUNNER_PAIR",
                         "buggy_version": buggy["scipy"], "fixed_version": fixed["scipy"]})


def run(test_id: str, contract_path: Path, case_path: Path | None = None,
        runner: Path | None = None, legacy: Path | None = None,
        current: Path | None = None) -> dict:
    if test_id not in {"reference.reference_solution", "property.representation_equivalence", "replay.version_pair"}:
        raise ValueError("unsupported test_id")
    try:
        _, relation, contract_hash = _contract(contract_path, test_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {"reason": "INVALID_CONTRACT", "error": str(error)})
    if test_id != "replay.version_pair":
        return _current_relation(test_id, relation, contract_hash)
    if None in (case_path, runner, legacy, current):
        return _observation(test_id, "NON_EVALUABLE", 0, {"reason": "MISSING_REPLAY_ARGUMENT"})
    return _pair_relation(test_id, case_path, runner, legacy, current, contract_hash)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--legacy-python", type=Path)
    parser.add_argument("--current-python", type=Path)
    args = parser.parse_args()
    result = run(args.test_id, args.contract, args.case, args.runner,
                 args.legacy_python, args.current_python)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

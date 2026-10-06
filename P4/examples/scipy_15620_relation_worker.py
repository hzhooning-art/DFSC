"""Live reference, dtype, and historical-pair relations for SciPy #15620.

The recorded float64 vector is a version-stable fixture, not an independent
mathematical oracle for the absolute correctness of resample_poly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


CASE_ID = "scipy_15620"
SCHEMA = "P4-Relation-Observation-v1"
IDS = {
    "reference.float64_reference": ("reference", "finite_signal_float64", "allclose_to_recorded_reference"),
    "property.dtype_consistency": ("property", "lossless_integer_casts", "allclose_to_float64_reference"),
    "replay.version_pair": ("replay", "unchanged_runner_pair", "paired_role_semantics"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(test_id: str, outcome: str, calls: int, details: dict) -> dict:
    return {"schema": SCHEMA, "test_id": test_id, "outcome": outcome,
            "subject_invocations": calls, "details": details}


def _close(actual: list, expected: list, rtol: float, atol: float) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    return all(isinstance(a, (int, float)) and not isinstance(a, bool) and
               math.isfinite(a) and math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
               for a, b in zip(actual, expected))


def _version(raw: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in raw.split("."))
    if not parts:
        raise ValueError("empty version")
    return parts


def _inputs(contract_path: Path, fixture_path: Path, test_id: str) -> tuple[dict, dict, dict, str, str]:
    contract_source = contract_path.read_bytes()
    fixture_source = fixture_path.read_bytes()
    contract = json.loads(contract_source)
    fixture = json.loads(fixture_source)
    if contract.get("contract_id") != "scipy-resample-poly-dtype-v1":
        raise ValueError("wrong contract_id")
    relation = contract["relations"][test_id.split(".", 1)[1]]
    if (relation.get("layer"), relation.get("generator"), relation.get("comparator")) != IDS[test_id]:
        raise ValueError("relation declaration mismatch")
    if (fixture.get("schema") != "P4-Recorded-Reference-Fixture-v1" or
            fixture.get("case_id") != CASE_ID or fixture.get("dtype") != "float64" or
            fixture.get("input") != [0, 1, 2, 3, 2, 1, 0] or
            (fixture.get("up"), fixture.get("down"), fixture.get("padtype")) != (2, 1, "smooth") or
            contract["domain"]["parameters"] != {"up": [2], "down": [1], "padtype": ["smooth"]} or
            contract["execution"]["dtypes"] != ["int16", "int32", "float32", "float64"] or
            contract["references"]["reference_dtype"] != "float64"):
        raise ValueError("fixture and contract domain mismatch")
    output = fixture.get("output")
    if (not isinstance(output, list) or len(output) != 14 or
            not all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                    math.isfinite(value) for value in output) or max(abs(value) for value in output) < 1.0):
        raise ValueError("invalid or degenerate recorded reference")
    if test_id != "replay.version_pair":
        for key in ("rtol", "atol"):
            value = relation["parameters"][key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid {key}")
    return (contract, relation, fixture, hashlib.sha256(contract_source).hexdigest(),
            hashlib.sha256(fixture_source).hexdigest())


def _current_paths() -> None:
    site = os.environ.get("P4_SCIPY_SITE_PACKAGES")
    dll = os.environ.get("P4_SCIPY_DLL_DIR")
    if site:
        sys.path.insert(0, site)
    if dll and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll)


def _current_relation(test_id: str, relation: dict, fixture: dict,
                      contract_hash: str, fixture_hash: str) -> dict:
    try:
        _current_paths()
        import numpy as np
        import scipy
        from scipy.signal import resample_poly
    except (ImportError, OSError, ValueError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {"reason": "SCIPY_UNAVAILABLE", "error": str(error)})
    dtypes = ["float64"] if test_id == "reference.float64_reference" else ["float64", "int16", "int32"]
    observed: dict[str, list] = {}
    for dtype in dtypes:
        try:
            values = np.asarray(fixture["input"], dtype=dtype)
            result = np.asarray(resample_poly(values, fixture["up"], fixture["down"],
                                              padtype=fixture["padtype"]))
            observed[dtype] = result.astype(np.float64).tolist()
        except Exception as error:
            return _observation(test_id, "FAIL", len(observed) + 1,
                                {"reason": "SUBJECT_EXCEPTION", "dtype": dtype,
                                 "error": type(error).__name__, "message": str(error),
                                 "prior_observations": observed, "scipy_version": scipy.__version__})
    rtol, atol = relation["parameters"]["rtol"], relation["parameters"]["atol"]
    reference_ok = _close(observed["float64"], fixture["output"], 1e-12, 1e-12)
    details = {"observed": observed, "recorded_reference": fixture["output"],
               "recorded_reference_is_independent_oracle": False,
               "contract_sha256": contract_hash, "fixture_sha256": fixture_hash,
               "scipy_version": scipy.__version__, "scipy_file": str(Path(scipy.__file__).resolve()),
               "python_executable": sys.executable, "rtol": rtol, "atol": atol,
               "historical_pair_executed": False}
    if not reference_ok:
        return _observation(test_id, "FAIL" if test_id == "reference.float64_reference" else "NON_EVALUABLE",
                            len(dtypes), {**details, "reason": "FLOAT64_REFERENCE_MISMATCH"})
    passed = test_id == "reference.float64_reference" or all(
        _close(observed[dtype], observed["float64"], rtol, atol) for dtype in ("int16", "int32")
    )
    return _observation(test_id, "PASS" if passed else "FAIL", len(dtypes), details)


def _run_role(python: Path, runner: Path, role: str, env: dict[str, str]) -> dict:
    argv = [str(python), "-B", str(runner), "--expected-role", role]
    try:
        result = subprocess.run(argv, capture_output=True, check=False,
                                timeout=8.0, env=env, shell=False)
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
    if (not isinstance(payload, dict) or
            payload.get("schema") != "P4-SciPy-Resample-Poly-Historical-Pair-Runner-v1" or
            payload.get("case_id") != CASE_ID or payload.get("expected_role") != role or
            not isinstance(payload.get("observation"), dict)):
        record["status"] = "INVALID_OBSERVATION"
        return record
    record["payload"] = payload
    return record


def _role_valid(payload: dict, fixture: dict, fixed: bool) -> bool:
    obs = payload["observation"]
    if (payload["role_confirmed"] is not True or payload["parameters"] !=
            {"input": fixture["input"], "up": 2, "down": 1, "padtype": "smooth"}):
        return False
    by_dtype = obs["by_dtype"]
    if set(by_dtype) != {"int16", "int32", "float32", "float64"}:
        return False
    for dtype, item in by_dtype.items():
        values = item["output"]
        if (item["dtype"] != dtype or not isinstance(values, list) or len(values) != 14 or
                any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) for value in values) or
                item["nonzero_count"] != sum(value != 0 for value in values) or
                not math.isclose(item["max_abs"], max(abs(value) for value in values),
                                 rel_tol=0.0, abs_tol=1e-12)):
            return False
    reference = by_dtype["float64"]["output"]
    if not _close(reference, fixture["output"], 1e-12, 1e-12):
        return False
    integers = (by_dtype["int16"]["output"], by_dtype["int32"]["output"])
    if fixed:
        return (obs["matches_fixed_expectation"] is True and
                all(_close(values, reference, 1e-12, 1e-12) for values in integers))
    return (obs["matches_reported_bug"] is True and
            all(all(value == 0.0 for value in values) for values in integers) and
            any(value != 0.0 for value in reference))


def _pair_relation(test_id: str, case_path: Path, runner: Path, legacy: Path,
                   current: Path, fixture: dict, contract_hash: str, fixture_hash: str) -> dict:
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if (case.get("case_id") != CASE_ID or
                runner.resolve() != (Path(__file__).resolve().parents[1] / case["runner_path"]).resolve()):
            raise ValueError("frozen case/runner path mismatch")
        runner_hash = _sha256(runner)
        if runner_hash != case["runner_sha256"]:
            raise ValueError("frozen runner SHA-256 mismatch")
        if not legacy.is_file() or not current.is_file():
            raise ValueError("one or both interpreters are missing")
        hashes = {"runner_sha256": runner_hash, "legacy_python_sha256": _sha256(legacy),
                  "current_python_sha256": _sha256(current),
                  "contract_sha256": contract_hash, "fixture_sha256": fixture_hash}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {"reason": "INVALID_OR_MISSING_REPLAY_INPUT", "error": str(error)})
    old_env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "P4_SCIPY_SITE_PACKAGES", "P4_SCIPY_DLL_DIR"):
        old_env.pop(name, None)
    new_env = os.environ.copy()
    site = new_env.get("P4_SCIPY_SITE_PACKAGES")
    if site:
        fixture_dir = Path(__file__).resolve().parents[1] / "tests/reconstruction/fixtures"
        new_env["PYTHONPATH"] = os.pathsep.join((str(fixture_dir), site))
    old = _run_role(legacy, runner, "buggy", old_env)
    new = _run_role(current, runner, "fixed", new_env)
    details = {**hashes, "buggy": old, "fixed": new, "same_runner_for_both_roles": True,
               "captured_pair_file_used_as_observation": False,
               "recorded_reference_is_independent_oracle": False}
    if old["status"] != "OBSERVED" or new["status"] != "OBSERVED":
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {**details, "reason": "ROLE_EXECUTION_INCOMPLETE"})
    try:
        buggy, fixed = old["payload"], new["payload"]
        old_version, new_version = _version(buggy["scipy"]), _version(fixed["scipy"])
        old_ok = old_version == (1, 14, 1) and _role_valid(buggy, fixture, fixed=False)
        new_ok = new_version >= (1, 15, 0) and new_version > old_version and _role_valid(fixed, fixture, fixed=True)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _observation(test_id, "NON_EVALUABLE", 8,
                            {**details, "reason": "INVALID_ROLE_SEMANTICS", "error": str(error)})
    if not old_ok:
        return _observation(test_id, "NON_EVALUABLE", 8,
                            {**details, "reason": "BUGGY_ROLE_NOT_REPRODUCED"})
    return _observation(test_id, "PASS" if new_ok else "FAIL", 8,
                        {**details, "reason": "LIVE_UNCHANGED_RUNNER_PAIR",
                         "buggy_version": buggy["scipy"], "fixed_version": fixed["scipy"]})


def run(test_id: str, contract_path: Path, fixture_path: Path,
        case_path: Path | None = None, runner: Path | None = None,
        legacy: Path | None = None, current: Path | None = None) -> dict:
    if test_id not in IDS:
        raise ValueError("unsupported test_id")
    try:
        _, relation, fixture, contract_hash, fixture_hash = _inputs(contract_path, fixture_path, test_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation(test_id, "NON_EVALUABLE", 0,
                            {"reason": "INVALID_CONTRACT_OR_FIXTURE", "error": str(error)})
    if test_id != "replay.version_pair":
        return _current_relation(test_id, relation, fixture, contract_hash, fixture_hash)
    if any(item is None for item in (case_path, runner, legacy, current)):
        return _observation(test_id, "NON_EVALUABLE", 0, {"reason": "MISSING_REPLAY_ARGUMENT"})
    return _pair_relation(test_id, case_path, runner, legacy, current,
                          fixture, contract_hash, fixture_hash)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--legacy-python", type=Path)
    parser.add_argument("--current-python", type=Path)
    args = parser.parse_args()
    result = run(args.test_id, args.contract, args.fixture, args.case,
                 args.runner, args.legacy_python, args.current_python)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

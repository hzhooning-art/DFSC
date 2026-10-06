"""Check the SciPy #8928 normal-mean relation on the current wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


TEST_ID = "reference.analytic_mean"


def _observation(outcome: str, calls: int, details: dict) -> dict:
    return {"schema": "P4-Relation-Observation-v1", "test_id": TEST_ID,
            "outcome": outcome, "subject_invocations": calls, "details": details}


def _contract_inputs(path: Path) -> tuple[list[tuple[float, float]], float, str]:
    source = path.read_bytes()
    contract = json.loads(source)
    relation = contract["relations"]["analytic_mean"]
    if (contract.get("contract_id") != "scipy-normal-expect-location-v1" or
            (relation.get("layer"), relation.get("generator"), relation.get("comparator")) !=
            ("reference", "normal_location_scale_cases", "absolute_expectation_error") or
            "value_error" not in relation.get("mechanisms", ())):
        raise ValueError("unsupported contract or relation declaration")
    locations = contract["domain"]["parameters"]["loc"]
    scales = contract["domain"]["parameters"]["scale"]
    if (not isinstance(locations, list) or not isinstance(scales, list) or
            len(locations) != 2 or len(scales) != 2):
        raise ValueError("expected two paired location-scale cases")
    cases = []
    for loc, scale in zip(locations, scales):
        if (not isinstance(loc, (int, float)) or isinstance(loc, bool) or
                not isinstance(scale, (int, float)) or isinstance(scale, bool) or
                not math.isfinite(loc) or not math.isfinite(scale) or scale <= 0):
            raise ValueError("invalid location-scale case")
        cases.append((float(loc), float(scale)))
    tolerance = relation["parameters"]["atol"]
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or
            not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("invalid absolute tolerance")
    return cases, float(tolerance), hashlib.sha256(source).hexdigest()


def run(contract_path: Path) -> dict:
    try:
        cases, tolerance, contract_hash = _contract_inputs(contract_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation("NON_EVALUABLE", 0, {"reason": "INVALID_CONTRACT", "error": str(error)})
    site = os.environ.get("P4_SCIPY_SITE_PACKAGES")
    dll = os.environ.get("P4_SCIPY_DLL_DIR")
    if site:
        sys.path.insert(0, site)
    if dll and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll)
    try:
        import scipy
        from scipy import stats
    except (ImportError, OSError, ValueError) as error:
        return _observation("NON_EVALUABLE", 0, {"reason": "SCIPY_UNAVAILABLE", "error": str(error)})
    checks = []
    for loc, scale in cases:
        try:
            observed = float(stats.norm.expect(loc=loc, scale=scale))
        except (ArithmeticError, TypeError, ValueError) as error:
            return _observation("FAIL", len(checks) + 1,
                                {"reason": "SUBJECT_EXCEPTION", "error": str(error),
                                 "scipy_version": scipy.__version__, "prior_checks": checks})
        checks.append({"loc": loc, "scale": scale, "observed": observed,
                       "analytic_reference": loc, "absolute_error": abs(observed - loc)})
    details = {"checks": checks, "atol": tolerance, "reference_method": "analytic-normal-mean",
               "scipy_version": scipy.__version__, "scipy_file": str(Path(scipy.__file__).resolve()),
               "python_executable": sys.executable, "contract_sha256": contract_hash,
               "historical_pair_executed": False}
    passed = all(math.isfinite(row["observed"]) and row["absolute_error"] <= tolerance
                 for row in checks)
    return _observation("PASS" if passed else "FAIL", len(checks), details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    if args.test_id != TEST_ID:
        raise SystemExit(f"unsupported test_id: {args.test_id}")
    print(json.dumps(run(args.contract), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

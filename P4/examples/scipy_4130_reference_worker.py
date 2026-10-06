"""Execute the declared SciPy #4130 value relation against an installed wheel.

This worker checks one current-version subject call against a separate
integer-hypergeometric enumeration. It does not run a historical version pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


TEST_ID = "reference.exact_enumeration"
OBSERVATION_SCHEMA = "P4-Relation-Observation-v1"


def _observation(outcome: str, calls: int, details: dict) -> dict:
    return {"schema": OBSERVATION_SCHEMA, "test_id": TEST_ID,
            "outcome": outcome, "subject_invocations": calls, "details": details}


def _table_and_tolerance(contract_path: Path) -> tuple[tuple[tuple[int, int], tuple[int, int]], float, str]:
    source = contract_path.read_bytes()
    contract = json.loads(source)
    if contract.get("contract_id") != "scipy-fisher-exact-two-sided-v1":
        raise ValueError("wrong contract_id")
    relation = contract["relations"]["exact_enumeration"]
    if (relation.get("layer"), relation.get("generator"), relation.get("comparator")) != (
        "reference", "contingency_margin_2x2_table", "absolute_pvalue_error"
    ) or "value_error" not in relation.get("mechanisms", ()):
        raise ValueError("unsupported relation declaration")
    entries = contract["domain"]["parameters"]["table"]
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("expected exactly one 2x2 table")
    raw = entries[0]
    if (not isinstance(raw, list) or len(raw) != 2 or
            any(not isinstance(row, list) or len(row) != 2 for row in raw)):
        raise ValueError("invalid 2x2 table")
    cells = [cell for row in raw for cell in row]
    if any(not isinstance(cell, int) or isinstance(cell, bool) or cell < 0 for cell in cells):
        raise ValueError("table cells must be nonnegative integers")
    table = ((cells[0], cells[1]), (cells[2], cells[3]))
    tolerance = relation["parameters"]["atol"]
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("atol must be finite and positive")
    return table, float(tolerance), hashlib.sha256(source).hexdigest()


def exact_two_sided_pvalue(table: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """Enumerate all tables with the same margins using integer weights."""

    (a, b), (c, d) = table
    row0, row1, col0 = a + b, c + d, a + c
    if not row0 or not row1 or not col0 or col0 == row0 + row1:
        raise ValueError("degenerate table margins")
    lower, upper = max(0, col0 - row1), min(row0, col0)
    observed_weight = math.comb(row0, a) * math.comb(row1, col0 - a)
    numerator = sum(
        weight for x in range(lower, upper + 1)
        if (weight := math.comb(row0, x) * math.comb(row1, col0 - x)) <= observed_weight
    )
    return numerator / math.comb(row0 + row1, col0)


def _add_local_scipy_paths() -> None:
    site = os.environ.get("P4_SCIPY_SITE_PACKAGES")
    dll = os.environ.get("P4_SCIPY_DLL_DIR")
    if site:
        sys.path.insert(0, site)
    if dll and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll)


def run(contract_path: Path) -> dict:
    try:
        table, tolerance, contract_sha256 = _table_and_tolerance(contract_path)
        reference = exact_two_sided_pvalue(table)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _observation("NON_EVALUABLE", 0, {"reason": "INVALID_CONTRACT_OR_ORACLE", "error": str(error)})
    try:
        _add_local_scipy_paths()
        import scipy
        from scipy import stats
    except (ImportError, OSError, ValueError) as error:
        return _observation("NON_EVALUABLE", 0, {"reason": "SCIPY_UNAVAILABLE", "error": str(error)})
    try:
        observed = float(stats.fisher_exact(table, alternative="two-sided").pvalue)
    except (ArithmeticError, TypeError, ValueError) as error:
        return _observation("FAIL", 1, {"reason": "SUBJECT_EXCEPTION", "error": str(error),
                                        "scipy_version": scipy.__version__})
    error = abs(observed - reference)
    details = {
        "table": table,
        "observed_pvalue": observed,
        "reference_pvalue": reference,
        "absolute_pvalue_error": error,
        "atol": tolerance,
        "reference_method": "integer-hypergeometric-enumeration",
        "scipy_version": scipy.__version__,
        "scipy_file": str(Path(scipy.__file__).resolve()),
        "python_executable": sys.executable,
        "contract_sha256": contract_sha256,
        "historical_pair_executed": False,
    }
    outcome = "PASS" if math.isfinite(observed) and error <= tolerance else "FAIL"
    return _observation(outcome, 1, details)


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

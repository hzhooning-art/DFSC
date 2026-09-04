"""Dependency-minimal cross-version runner for SciPy issue #8906."""

from __future__ import print_function

import argparse
import json
import sys

import numpy as np
import scipy
from scipy.linalg import solve_banded


def attempt(bands, matrix, rhs):
    try:
        solution = solve_banded(bands, matrix, rhs.copy())
        return {"executed": True, "exception": None, "solution": np.asarray(solution).tolist()}
    except Exception as error:  # exact exception is evidence on the buggy side
        return {"executed": False, "exception": type(error).__name__, "message": str(error)}


def run(expected_role):
    rhs = np.asarray([[1.0, 2.0, 3.0]])
    expected = np.asarray([[0.5, 1.0, 1.5]])
    conventional = attempt((0, 0), np.asarray([[2.0]]), rhs)
    padded = attempt((1, 1), np.asarray([[0.0], [2.0], [0.0]]), rhs)
    conventional_pass = conventional["executed"] and np.allclose(conventional["solution"], expected)
    padded_pass = padded["executed"] and np.allclose(padded["solution"], expected)
    matches_reported_bug = (not conventional["executed"] or not conventional_pass) and padded_pass
    matches_fixed_expectation = conventional_pass and padded_pass
    role_confirmed = matches_reported_bug if expected_role == "buggy" else matches_fixed_expectation
    return {
        "schema": "P4-SciPy-Historical-Pair-Runner-v1",
        "case_id": "scipy_8906",
        "expected_role": expected_role,
        "role_confirmed": bool(role_confirmed),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "upstream_issue": "https://github.com/scipy/scipy/issues/8906",
        "reported_buggy_release": "1.14.1",
        "first_fixed_release": "1.15.0",
        "observation": {
            "conventional_u0": conventional,
            "padded_u1": padded,
            "matches_reported_bug": bool(matches_reported_bug),
            "matches_fixed_expectation": bool(matches_fixed_expectation),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-role", choices=("buggy", "fixed"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.expected_role), indent=2))


if __name__ == "__main__":
    main()

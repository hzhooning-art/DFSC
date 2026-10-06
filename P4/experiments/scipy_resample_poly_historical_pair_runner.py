"""Dependency-minimal cross-version runner for SciPy issue #15620."""

from __future__ import print_function

import argparse
import json
import sys

import numpy as np
import scipy
from scipy.signal import resample_poly


INPUT = [0, 1, 2, 3, 2, 1, 0]


def evaluate(dtype):
    values = np.asarray(INPUT, dtype=dtype)
    output = np.asarray(resample_poly(values, 2, 1, padtype="smooth"))
    return {
        "dtype": str(values.dtype),
        "output_dtype": str(output.dtype),
        "output": output.tolist(),
        "nonzero_count": int(np.count_nonzero(output)),
        "max_abs": float(np.max(np.abs(output))),
    }


def run(expected_role):
    observations = {name: evaluate(name) for name in ("int16", "int32", "float32", "float64")}
    reference = np.asarray(observations["float64"]["output"], dtype=np.float64)
    integer_errors = {}
    for name in ("int16", "int32"):
        output = np.asarray(observations[name]["output"], dtype=np.float64)
        integer_errors[name] = float(np.max(np.abs(output - reference)))

    float_reference_nonzero = observations["float64"]["nonzero_count"] > 0
    integer_outputs_all_zero = all(observations[name]["nonzero_count"] == 0 for name in ("int16", "int32"))
    integer_outputs_match_reference = all(
        np.allclose(observations[name]["output"], reference, rtol=1e-12, atol=1e-12)
        for name in ("int16", "int32")
    )
    matches_reported_bug = float_reference_nonzero and integer_outputs_all_zero
    matches_fixed_expectation = float_reference_nonzero and integer_outputs_match_reference
    role_confirmed = matches_reported_bug if expected_role == "buggy" else matches_fixed_expectation

    return {
        "schema": "P4-SciPy-Resample-Poly-Historical-Pair-Runner-v1",
        "case_id": "scipy_15620",
        "expected_role": expected_role,
        "role_confirmed": bool(role_confirmed),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "upstream_issue": "https://github.com/scipy/scipy/issues/15620",
        "reported_buggy_release": "1.14.1",
        "first_fixed_release": "1.15.0",
        "parameters": {"input": INPUT, "up": 2, "down": 1, "padtype": "smooth"},
        "observation": {
            "by_dtype": observations,
            "integer_max_abs_error_vs_float64": integer_errors,
            "float_reference_nonzero": bool(float_reference_nonzero),
            "integer_outputs_all_zero": bool(integer_outputs_all_zero),
            "integer_outputs_match_reference": bool(integer_outputs_match_reference),
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

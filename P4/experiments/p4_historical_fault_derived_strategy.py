"""Stage 6 strategy replay derived from the SciPy #8906 historical defect.

This is deliberately not an old-version execution.  The compact faulty
function reproduces the reported hard-coded band-row selection so that a weak
example and a representation-equivalence property can be compared on the same
case family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import solve_banded


ISSUE_URL = "https://github.com/scipy/scipy/issues/8906"
REGRESSION_SOURCE = Path(scipy.__file__).resolve().parent / "linalg" / "tests" / "test_basic.py"
TRIALS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reported_fault_surrogate(bands: tuple[int, int], ab, b) -> np.ndarray:
    """Reproduce the reported 1x1 special case that always selected row one."""
    del bands
    band_array = np.asarray(ab, dtype=float)
    rhs = np.asarray(b, dtype=float).copy()
    return rhs / band_array[1, 0]


def _call(solver, bands, ab, b) -> dict:
    try:
        value = np.asarray(solver(bands, ab, b), dtype=float)
        return {"returned": True, "value": value.tolist(), "exception": None}
    except Exception as exc:  # the exception is the historical observable
        return {"returned": False, "value": None, "exception": type(exc).__name__}


def evaluate_trial(index: int) -> dict:
    scale = 0.25 + 0.125 * index
    rhs = np.array([[scale, -2.0 * scale, 3.0 * scale]])
    expected = rhs / 2.0
    conventional = ((0, 0), [[2.0]])
    padded = ((1, 1), [[0.0], [2.0], [0.0]])

    buggy_padded = _call(reported_fault_surrogate, *padded, rhs)
    buggy_conventional = _call(reported_fault_surrogate, *conventional, rhs)
    fixed_padded = _call(solve_banded, *padded, rhs)
    fixed_conventional = _call(solve_banded, *conventional, rhs)

    weak_example_accepts_bug = (
        buggy_padded["returned"]
        and np.allclose(buggy_padded["value"], expected, rtol=1.0e-15, atol=0.0)
    )
    property_detects_bug = not (
        buggy_conventional["returned"]
        and np.allclose(buggy_conventional["value"], buggy_padded["value"], rtol=1.0e-15, atol=0.0)
    )
    fixed_satisfies_property = (
        fixed_padded["returned"]
        and fixed_conventional["returned"]
        and np.allclose(fixed_padded["value"], fixed_conventional["value"], rtol=1.0e-15, atol=0.0)
        and np.allclose(fixed_conventional["value"], expected, rtol=1.0e-15, atol=0.0)
    )
    return {
        "trial": index,
        "rhs_scale": scale,
        "weak_padded_example_detected_fault": not weak_example_accepts_bug,
        "representation_equivalence_detected_fault": property_detects_bug,
        "current_fixed_satisfies_equivalence": fixed_satisfies_property,
        "faulty_observations": {
            "padded": buggy_padded,
            "conventional": buggy_conventional,
        },
    }


def run() -> dict:
    records = [evaluate_trial(index) for index in range(TRIALS)]
    return {
        "schema": "P4-Historical-Fault-Derived-Strategy-v1",
        "case": {
            "project": "scipy/scipy",
            "issue": 8906,
            "issue_url": ISSUE_URL,
            "installed_fixed_version": scipy.__version__,
            "installed_regression_source_sha256": sha256(REGRESSION_SOURCE),
            "fault_construction": "source-derived compact surrogate of the reported hard-coded row-one selection",
            "old_buggy_package_executed": False,
        },
        "summary": {
            "paired_input_variants": len(records),
            "weak_example_detections": sum(row["weak_padded_example_detected_fault"] for row in records),
            "equivalence_property_detections": sum(row["representation_equivalence_detected_fault"] for row in records),
            "fixed_control_passes": sum(row["current_fixed_satisfies_equivalence"] for row in records),
            "independent_historical_defect_families": 1,
            "complete_buggy_fixed_environment_pairs": 0,
        },
        "records": records,
        "claim_boundary": (
            "The 32 inputs are paired variants of one upstream defect family, not 32 independent defects. "
            "The faulty implementation is a source-derived surrogate, not execution of a historical SciPy "
            "release; this result measures strategy discrimination and does not create a complete buggy/fixed pair."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()

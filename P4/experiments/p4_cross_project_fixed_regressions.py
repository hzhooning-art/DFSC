"""Stage 5 cross-project capture of current fixed-side regression cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import solve_banded


ROOT = Path(__file__).resolve().parents[1]
PYTORCH_CAPTURE = ROOT / "results" / "p4_historical_fixed_replays.json"
SCIPY_REGRESSION_SOURCE = Path(scipy.__file__).resolve().parent / "linalg" / "tests" / "test_basic.py"
SCIPY_ISSUE = "https://github.com/scipy/scipy/issues/8906"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_scipy_8906() -> dict:
    right_hand_side = np.array([[1.0, 2.0, 3.0]])
    original = right_hand_side.copy()
    expected = np.array([[0.5, 1.0, 1.5]])
    conventional = solve_banded((0, 0), [[2.0]], right_hand_side)
    padded = solve_banded((1, 1), [[0.0], [2.0], [0.0]], right_hand_side)
    conventional_error = float(np.max(np.abs(conventional - expected)))
    padded_error = float(np.max(np.abs(padded - expected)))
    input_unchanged = bool(np.array_equal(right_hand_side, original))
    tolerance = 1.0e-14
    role_confirmed = conventional_error <= tolerance and padded_error <= tolerance and input_unchanged
    return {
        "case_id": "scipy-8906",
        "project": "scipy/scipy",
        "issue_url": SCIPY_ISSUE,
        "reported_defect": "solve_banded selected the wrong band row for a 1x1 system",
        "role": "fixed",
        "environment": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "observation": {
            "executed": True,
            "conventional_max_abs_error": conventional_error,
            "padded_band_max_abs_error": padded_error,
            "right_hand_side_unchanged": input_unchanged,
            "tolerance": tolerance,
        },
        "role_confirmed": role_confirmed,
    }


def run() -> dict:
    pytorch = json.loads(PYTORCH_CAPTURE.read_text(encoding="utf-8"))
    pytorch_records = [
        {**row, "project": "pytorch/pytorch"}
        for row in pytorch["records"]
    ]
    records = [*pytorch_records, replay_scipy_8906()]
    projects = sorted({row["project"] for row in records})
    return {
        "schema": "P4-Cross-Project-Fixed-Regressions-v1",
        "provenance": {
            "pytorch_capture": str(PYTORCH_CAPTURE.relative_to(ROOT)),
            "pytorch_runner_sha256": pytorch["runner_sha256"],
            "scipy_regression_source": str(Path("..") / SCIPY_REGRESSION_SOURCE.relative_to(ROOT.parent)),
            "scipy_regression_source_sha256": sha256(SCIPY_REGRESSION_SOURCE),
            "scipy_issue_url": SCIPY_ISSUE,
        },
        "records": records,
        "summary": {
            "projects": projects,
            "project_count": len(projects),
            "attempted": len(records),
            "executed": sum(row["observation"]["executed"] for row in records),
            "fixed_roles_confirmed": sum(row["role_confirmed"] for row in records),
            "buggy_roles_confirmed": 0,
            "complete_pairs": 0,
        },
        "claim_boundary": (
            "This stage broadens only current fixed-side project coverage. The SciPy replay is "
            "derived from an installed upstream regression test; without execution in a reported "
            "buggy release, it is not a complete historical buggy/fixed pair or a detection result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()

"""Role-independent reproducer for SciPy issue #8928."""

from __future__ import annotations

import json
import math

import scipy
from scipy import stats


CASES = ((36.0, 1.0), (10.0, 0.1))


def main():
    checks = []
    for loc, scale in CASES:
        observed = float(stats.norm.expect(loc=loc, scale=scale))
        checks.append({
            "loc": loc,
            "scale": scale,
            "observed": observed,
            "analytic_reference": loc,
            "absolute_error": abs(observed - loc),
        })
    max_error = max(item["absolute_error"] for item in checks)
    passed = all(math.isfinite(item["observed"]) for item in checks) and max_error < 1e-10
    print(json.dumps({
        "schema": "P4-Role-Independent-Runner-v1",
        "case_id": "scipy_8928",
        "package": "scipy",
        "version": scipy.__version__,
        "outcome": "pass" if passed else "fail",
        "invocations": len(checks),
        "details": {
            "checks": checks,
            "max_absolute_expectation_error": max_error,
            "reference_method": "analytic-normal-mean",
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()

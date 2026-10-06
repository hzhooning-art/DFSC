"""Role-independent reproducer for SciPy issue #4130."""

from __future__ import annotations

import json
import math

import scipy
from scipy import stats


TABLE = ((345, 455), (260, 345))


def exact_two_sided_pvalue(table):
    """Enumerate the conditional 2x2 distribution with integer weights."""
    (a, b), (c, d) = table
    row0 = a + b
    row1 = c + d
    col0 = a + c
    total = row0 + row1
    lower = max(0, col0 - row1)
    upper = min(row0, col0)
    observed_weight = math.comb(row0, a) * math.comb(row1, col0 - a)
    numerator = sum(
        weight
        for x in range(lower, upper + 1)
        if (weight := math.comb(row0, x) * math.comb(row1, col0 - x))
        <= observed_weight
    )
    return numerator / math.comb(total, col0)


def main():
    # Tuple indexing is stable across the historical and current SciPy APIs.
    observed = float(stats.fisher_exact(TABLE)[1])
    reference = exact_two_sided_pvalue(TABLE)
    error = abs(observed - reference)
    passed = math.isfinite(observed) and error < 1e-12
    print(json.dumps({
        "schema": "P4-Role-Independent-Runner-v1",
        "case_id": "scipy_4130",
        "package": "scipy",
        "version": scipy.__version__,
        "outcome": "pass" if passed else "fail",
        "invocations": 2,
        "details": {
            "table": TABLE,
            "observed_pvalue": observed,
            "reference_pvalue": reference,
            "absolute_pvalue_error": error,
            "reference_method": "integer-hypergeometric-enumeration",
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()

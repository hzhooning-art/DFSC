"""Dependency-free execution example, not a historical-defect benchmark case."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, localcontext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--bias", type=float, default=0.0)
    args = parser.parse_args()
    points = (0.125, 0.5)
    observations = []
    if args.test_id == "reference.decimal_value":
        with localcontext() as context:
            context.prec = 50
            for x in points:
                observed = math.expm1(x) + args.bias
                reference = float(Decimal(str(x)).exp() - Decimal(1))
                observations.append({"x": x, "observed": observed, "reference": reference,
                                     "absolute_error": abs(observed - reference)})
        threshold = 1e-12
        calls = len(points)
    elif args.test_id == "property.central_derivative":
        step = 1e-5
        with localcontext() as context:
            context.prec = 50
            for x in points:
                derivative = (math.expm1(x + step) - math.expm1(x - step)) / (2 * step)
                reference = float(Decimal(str(x)).exp())
                observations.append({"x": x, "observed": derivative, "reference": reference,
                                     "absolute_error": abs(derivative - reference)})
        threshold = 1e-7
        calls = 2 * len(points)
    else:
        raise SystemExit(f"unsupported test_id: {args.test_id}")
    maximum = max(item["absolute_error"] for item in observations)
    print(json.dumps({
        "schema": "P4-Relation-Observation-v1",
        "test_id": args.test_id,
        "outcome": "PASS" if math.isfinite(maximum) and maximum <= threshold else "FAIL",
        "subject_invocations": calls,
        "details": {"points": observations, "threshold": threshold, "max_absolute_error": maximum},
    }, allow_nan=False))


if __name__ == "__main__":
    main()

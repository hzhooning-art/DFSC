"""Transfer example for a current SciPy API; not a historical defect case."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluate(contract: dict, test_id: str, bias: float = 0.0) -> dict:
    import numpy as np
    import scipy
    from scipy.integrate import solve_ivp

    relation = test_id.split(".", 1)[1]
    declared = contract["relations"][relation]
    if test_id != declared["layer"] + "." + relation:
        raise ValueError("relation binding mismatch")
    tolerance = declared["parameters"]["absolute_tolerance"]
    rows, calls = [], 0
    for rate in contract["domain"]["rates"]:
        def solve(initial: float):
            return solve_ivp(lambda t, y: -rate * y, (0.0, 2.0), [initial],
                             method="DOP853", t_eval=np.linspace(0, 2, 17),
                             rtol=1e-10, atol=1e-12)
        if relation == "trajectory":
            for initial in contract["domain"]["initial_values"]:
                result = solve(initial)
                calls += 1
                if not result.success:
                    raise RuntimeError(result.message)
                error = np.max(np.abs(result.y[0] + bias - initial * np.exp(-rate * result.t)))
                rows.append({"rate": rate, "initial": initial, "error": float(error), "nfev": result.nfev})
        elif relation == "amplitude_scaling":
            first, scaled = solve(1.0), solve(2.0)
            calls += 2
            if not first.success or not scaled.success:
                raise RuntimeError("integration did not succeed")
            rows.append({"rate": rate, "error": float(np.max(np.abs(scaled.y[0] - 2 * first.y[0])))})
        elif relation == "half_life_event":
            def event(t, y):
                return y[0] - 0.5
            event.terminal, event.direction = True, -1
            result = solve_ivp(lambda t, y: -rate * y, (0, 5 / rate), [1.0],
                               method="DOP853", events=event, rtol=1e-10, atol=1e-12)
            calls += 1
            if not result.success or len(result.t_events[0]) != 1:
                raise RuntimeError("expected one terminal event")
            observed, reference = float(result.t_events[0][0]), math.log(2) / rate
            rows.append({"rate": rate, "observed": observed, "reference": reference,
                         "error": abs(observed - reference)})
        else:
            raise ValueError("unsupported relation")
    maximum = max(row["error"] for row in rows)
    return {"schema": "P4-Relation-Observation-v1", "test_id": test_id,
            "outcome": "PASS" if math.isfinite(maximum) and maximum <= tolerance else "FAIL",
            "subject_invocations": calls,
            "details": {"rows": rows, "max_absolute_error": maximum,
                        "threshold": tolerance, "scipy_version": scipy.__version__,
                        "numpy_version": np.__version__,
                        "calls_mean": "reported solve_ivp calls, not RHS evaluations"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--bias", type=float, default=0.0)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(contract, args.test_id, args.bias), allow_nan=False))


if __name__ == "__main__":
    main()

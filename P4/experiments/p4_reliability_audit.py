"""Aggregate current P4 evidence into a machine-readable reliability audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "P4" / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    calibration = load("p4_integrated_calibration.json")
    reuse = load("p4_module_reuse_feasibility.json")
    operator = load("p4_operator_batch_reuse.json")
    physics = load("p4_physics_consistency_reuse.json")
    sweep = load("p4_physics_weight_sweep.json")

    gates = {
        "calibration_autograd": calibration["differentiable_calibration"]["gradient_finite"],
        "neural_module_reuse": all(row["gradient_finite"] for row in reuse["rows"]),
        "operator_batch_reuse": operator["direct_mlsl_operator"]["gradient_finite"],
        "physics_consistency_gradients": all(
            physics["summary"][name]["all_gradients_finite"]
            for name in ("data_only", "physics_consistent")
        ),
        "matched_weight_sweep_gradients": all(
            values["all_gradients_finite"] for values in sweep["summary"].values()
        ),
    }
    audit = {
        "protocol": "P4_PRIMITIVE_RELIABILITY_PROTOCOL",
        "status": "conformant" if all(gates.values()) else "nonconformant",
        "gates": gates,
        "scope_limits": [
            "known real stable Mittag-Leffler propagation family",
            "validated PyTorch CPU/GPU path",
            "no universal fractional-solver claim",
            "physics loss is task-dependent and trade-off controlled",
        ],
        "selected_physics_weight_by_rule": sweep["selected_weight"],
        "source_results": [
            "p4_integrated_calibration.json",
            "p4_module_reuse_feasibility.json",
            "p4_operator_batch_reuse.json",
            "p4_physics_consistency_reuse.json",
            "p4_physics_weight_sweep.json",
        ],
    }
    out = RESULTS / "p4_primitive_reliability_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

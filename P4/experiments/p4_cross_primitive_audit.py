"""Summarize protocol status across distinct primitive backends."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "P4" / "results"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    mlsl = read("p4_primitive_reliability_audit.json")
    smoke = read("p4_generic_protocol_smoke.json")
    matrix = read("p4_generic_matrix_exp_validation.json")
    matrix_long = read("p4_matrix_exp_ood_long_horizon.json")
    ode = read("p4_generic_ode_step_validation.json")
    ode_long = read("p4_rk4_ood_long_horizon.json")
    logistic = read("p4_nonlinear_ode_step_validation.json")
    backends = {
        "MLSL": {
            "status": mlsl["status"],
            "gates": mlsl["gates"],
            "domain": mlsl["scope_limits"],
        },
        "exponential_propagator_demo": {
            "status": smoke["status"],
            "gates": smoke["gates"],
            "domain": [smoke["domain"]["input_description"]],
        },
        "matrix_exponential_action": {
            "status": matrix["backend_audit"]["status"],
            "gates": matrix["backend_audit"]["gates"],
            "domain": [matrix["backend_audit"]["domain"]["input_description"]],
            "calibration": matrix["calibration"],
            "module_reuse": matrix["module_reuse"],
            "ood_long_horizon": matrix_long["summary"],
            "all_values_finite": matrix_long["all_values_finite"],
            "all_gradients_finite": matrix_long["all_gradients_finite"],
        },
        "rk4_linear_ode_step": {
            "status": ode["backend_audit"]["status"],
            "gates": ode["backend_audit"]["gates"],
            "domain": [ode["backend_audit"]["domain"]["input_description"]],
            "metrics": ode["backend_audit"]["metrics"],
            "calibration": ode["calibration"],
            "long_horizon": ode["long_horizon"],
            "module_reuse": ode["module_reuse"],
            "ood_long_horizon": ode_long["summary"],
            "all_ood_values_finite": ode_long["all_ood_values_finite"],
            "all_gradients_finite_multiseed": ode_long["all_gradients_finite"],
        },
        "logistic_rk4_step": {
            "status": "conformant" if all(
                (
                    logistic["all_ood_values_finite"],
                    logistic["all_gradients_finite"],
                    logistic["all_module_reuse_gradients_finite"],
                )
            ) else "nonconformant",
            "summary": logistic["summary"],
            "all_ood_values_finite": logistic["all_ood_values_finite"],
            "all_gradients_finite": logistic["all_gradients_finite"],
            "all_module_reuse_gradients_finite": logistic["all_module_reuse_gradients_finite"],
        },
    }
    result = {
        "protocol": "GENERAL_DIFFERENTIABLE_PRIMITIVE_PROTOCOL",
        "status": "cross_primitive_interface_supported",
        "backends": backends,
        "comparability_warning": (
            "Backend metrics are not pooled because the mathematical domains and "
            "reference problems differ; the evidence supports protocol reuse, "
            "not a universal accuracy ranking."
        ),
        "next_gate": "repeat calibration and module-level evaluation for the nonlinear ODE family before pooling any cross-backend scores",
    }
    out = RESULTS / "p4_cross_primitive_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

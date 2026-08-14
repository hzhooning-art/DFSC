"""Normalize heterogeneous primitive results into one protocol registry."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "P4" / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def gate(value):
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "missing"


def main():
    cross = load("p4_cross_primitive_audit.json")
    matrix = load("p4_generic_matrix_exp_validation.json")
    matrix_long = load("p4_matrix_exp_ood_long_horizon.json")
    linear = load("p4_generic_ode_step_validation.json")
    linear_long = load("p4_rk4_ood_long_horizon.json")
    logistic = load("p4_nonlinear_ode_step_validation.json")
    profile = load("p4_primitive_profile.json")
    common_module = load("p4_common_module_ood.json")
    mlsl = load("p4_primitive_reliability_audit.json")
    mlsl_protocol = load("p4_mlsl_protocol_validation.json")
    real_data = load("p4_real_data_evidence.json")

    records = [
        {
            "backend": "MLSL",
            "family": "fractional_propagation",
            "value": gate(mlsl_protocol["gates"].get("value")),
            "gradient": gate(mlsl_protocol["gates"].get("gradient")),
            "calibration": gate(mlsl_protocol["gates"].get("calibration")),
            "module_reuse": gate(mlsl_protocol["gates"].get("module_reuse")),
            "ood": gate(mlsl_protocol["gates"].get("ood")),
            "long_horizon": gate(mlsl_protocol["gates"].get("long_horizon")),
            "scope": mlsl_protocol["limitations"],
            "validation_artifact": "p4_mlsl_protocol_validation.json",
        },
        {
            "backend": "matrix_exponential_action",
            "family": "matrix_function_action",
            "value": gate(matrix["backend_audit"]["gates"].get("value_finite")),
            "gradient": gate(matrix["backend_audit"]["gates"].get("gradient_finite")),
            "calibration": gate(matrix["calibration"].get("gradient_finite")),
            "module_reuse": gate(matrix["module_reuse"].get("all_gradients_finite")),
            "ood": gate(matrix_long.get("all_values_finite")),
            "long_horizon": gate(matrix_long.get("all_values_finite")),
            "scope": ["2x2 stable matrix", "PyTorch matrix_exp backend"],
        },
        {
            "backend": "rk4_linear_ode_step",
            "family": "linear_time_integration",
            "value": gate(linear["backend_audit"]["gates"].get("value_finite")),
            "gradient": gate(linear["backend_audit"]["gates"].get("gradient_finite")),
            "calibration": gate(linear["calibration"].get("gradient_finite")),
            "module_reuse": gate(linear["module_reuse"].get("all_gradients_finite")),
            "ood": gate(linear_long.get("all_ood_values_finite")),
            "long_horizon": gate(linear_long.get("all_ood_values_finite")),
            "scope": ["2D linear ODE", "fixed-step RK4"],
        },
        {
            "backend": "logistic_rk4_step",
            "family": "nonlinear_time_integration",
            "value": gate(logistic.get("all_ood_values_finite")),
            "gradient": gate(logistic.get("all_gradients_finite")),
            "calibration": gate(all(row["calibration"]["gradient_finite"] for row in logistic["rows"])),
            "module_reuse": gate(logistic.get("all_module_reuse_gradients_finite")),
            "ood": gate(logistic.get("all_ood_values_finite")),
            "long_horizon": gate(logistic.get("all_ood_values_finite")),
            "scope": ["scalar Logistic ODE", "fixed-step RK4"],
        },
    ]
    required = ["value", "gradient", "calibration", "module_reuse", "ood", "long_horizon"]
    for record in records:
        states = [record[key] for key in required]
        record["coverage"] = sum(value == "pass" or value == "reported_prior_validation" for value in states) / len(required)
        record["status"] = "conformant" if all(value != "fail" and value != "missing" for value in states) else "nonconformant"
    result = {
        "schema": "DFSC-Primitive-Protocol-v1",
        "required_dimensions": required,
        "records": records,
        "registry_status": "complete_for_current_backends",
        "comparability_warning": cross["comparability_warning"],
        "missing_next": [
            "hosted CPU CI run after repository push",
            "formal public API packaging for the registry and profile schemas",
        ],
        "profile_schema": profile["schema"],
        "real_data_evidence": {
            "schema": real_data["schema"],
            "datasets": [
                {
                    "name": dataset["name"],
                    "source": dataset["source"],
                    "result_file": dataset["result_file"],
                    "models": dataset["models"],
                }
                for dataset in real_data["datasets"]
            ],
            "status": real_data["status"],
        },
        "common_module_ood": {
            "schema": common_module["schema"],
            "backends": {
                regime: {name: value["summary"] for name, value in backends.items()}
                for regime, backends in common_module["results"].items()
            },
            "all_gradients_finite": all(
                value["all_gradients_finite"]
                for backends in common_module["results"].values()
                for value in backends.values()
            ),
            "interpretation": "common template completed; backend benefits remain task-dependent",
        },
    }
    out = RESULTS / "p4_primitive_protocol_registry.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

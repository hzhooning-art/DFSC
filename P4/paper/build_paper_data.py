"""Build a small, auditable data bundle for the P4 manuscript tables.

The script reads experiment JSON files rather than duplicating measured values
in the manuscript source.  It intentionally preserves backend-specific metrics
and does not compute a pooled accuracy ranking.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "paper_data.json"


def read(name: str):
    with (RESULTS / name).open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    registry = read("p4_primitive_protocol_registry.json")
    profile = read("p4_primitive_profile.json")
    common = read("p4_common_module_ood.json")
    matrix = read("p4_matrix_exp_ood_long_horizon.json")
    rk4 = read("p4_rk4_ood_long_horizon.json")
    logistic = read("p4_nonlinear_ode_step_validation.json")
    simulation = read("p4_engineering_simulation_benchmark.json")
    calibration = read("p4_calibration_baselines.json")
    periodic_heat = read("p4_periodic_heat_2d_audit.json")
    matrix_conditioning = read("p4_matrix_ood_conditioning.json")
    precision_tradeoff = read("p4_precision_tradeoff.json")

    bundle = {
        "schema": "DFSC-P4-Paper-Data-v1",
        "source_files": [
            "results/p4_primitive_protocol_registry.json",
            "results/p4_primitive_profile.json",
            "results/p4_common_module_ood.json",
            "results/p4_matrix_exp_ood_long_horizon.json",
            "results/p4_rk4_ood_long_horizon.json",
            "results/p4_nonlinear_ode_step_validation.json",
            "results/p4_engineering_simulation_benchmark.json",
            "results/p4_calibration_baselines.json",
            "results/p4_periodic_heat_2d_audit.json",
            "results/p4_matrix_ood_conditioning.json",
            "results/p4_precision_tradeoff.json",
        ],
        "registry": registry,
        "profile": profile,
        "common_module_ood": common,
        "multi_seed": {
            "matrix_exponential_action": matrix,
            "rk4_linear_ode_step": rk4,
            "logistic_rk4_step": logistic,
        },
        "simulation_benchmark": simulation,
        "calibration_baselines": calibration,
        "periodic_heat_2d": periodic_heat,
        "matrix_ood_conditioning": matrix_conditioning,
        "precision_tradeoff": precision_tradeoff,
        "interpretation": (
            "The bundle supports protocol reuse and backend-specific reliability "
            "claims. It must not be used to claim a universal accuracy ranking."
        ),
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

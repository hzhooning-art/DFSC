"""Factorial audit of the nonmonotone Stage 62 decision boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from probe_public_pva_relaxation import (  # noqa: E402
    evaluate_grid_cell,
    fit_coefficients,
    fit_shared_rates,
    load_curves,
    resample,
)


OUTPUT = ROOT / "results" / "stage62_boundary_factor_audit.json"


def _diagnostics(curves, horizon: float, budget: int, starts: int = 3) -> dict:
    sampled = [resample(curve, horizon, budget) for curve in curves]
    fitted = fit_shared_rates(sampled, rank=3, starts=starts)
    rates = np.asarray(fitted["rates"])
    correlations = []
    for curve in sampled:
        _, prediction = fit_coefficients(curve.time, curve.value, rates)
        residual = curve.value - prediction
        if np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
            correlations.append(float(np.corrcoef(residual[:-1], residual[1:])[0, 1]))
    rho = float(np.median(correlations)) if correlations else 0.0
    effective_n = float(budget * (1.0 - np.clip(rho, -0.95, 0.95)) / (1.0 + np.clip(rho, -0.95, 0.95)))
    time = sampled[0].time
    sensitivity = np.column_stack([-time * np.exp(-rate * time) for rate in rates])
    singular = np.linalg.svd(sensitivity, compute_uv=False)
    return {
        "rates": rates.tolist(),
        "minimum_rate_ratio": fitted["minimum_rate_ratio"],
        "median_lag1_residual_correlation": rho,
        "ar1_effective_sample_size_proxy": effective_n,
        "rate_sensitivity_condition_number": float(singular[0] / max(singular[-1], 1e-15)),
    }


def _cell(curves, horizon: float, budget: int, starts: int = 3) -> dict:
    result = evaluate_grid_cell(curves, horizon, budget, starts=starts)
    result["factor_diagnostics"] = _diagnostics(curves, horizon, budget, starts=starts)
    return result


def main() -> None:
    curves = load_curves()
    tail = []
    for horizon in (4.0, 8.0, 15.0, 28.0):
        budget = max(12, int(round(96 * horizon / 28.0)))
        tail.append(_cell(curves, horizon, budget))
        print(f"tail horizon={horizon:g} budget={budget} {tail[-1]['decision']}", flush=True)
    density = []
    for budget in (12, 24, 48, 96):
        density.append(_cell(curves, 15.0, budget))
        print(f"density horizon=15 budget={budget} {density[-1]['decision']}", flush=True)
    optimizer = []
    for starts in (1, 2, 4, 8):
        result = evaluate_grid_cell(curves, 28.0, 96, starts=starts)
        optimizer.append({"starts": starts, "decision": result["decision"], "rank_records": result["rank_records"]})
        print(f"optimizer starts={starts} {result['decision']}", flush=True)
    payload = {
        "experiment": "stage62_nonmonotone_boundary_factor_audit",
        "tail_coverage_fixed_spacing": tail,
        "sampling_density_fixed_horizon": density,
        "optimizer_start_budget": optimizer,
        "interpretation_rule": (
            "Nonmonotonicity is attributed only when a controlled axis changes the gate outcome; "
            "residual correlation and sensitivity conditioning are diagnostics, not causal proofs."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

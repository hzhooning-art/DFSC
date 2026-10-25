"""Bounded response-only observability gate for P4's memory-kernel route.

The positive exponential spectrum is the same model class as the early P4
mixture prototypes. Unlike those prototypes, the observations here are causal
convolution responses, not samples of the hidden kernel. This is a best-case
finite-basis experiment, not a new estimator or a statistical guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize


def design_matrix(input_steps: np.ndarray, rates: np.ndarray, dt: float) -> np.ndarray:
    """Exact response for each unit-mass exponential to piecewise-constant input."""
    decay = np.exp(-rates * dt)
    state = np.zeros_like(rates)
    rows = np.zeros((input_steps.size + 1, rates.size))
    for index, value in enumerate(input_steps, start=1):
        state = decay * state + (1.0 - decay) * value
        rows[index] = state
    return rows


def fit_simplex(design: np.ndarray, observed: np.ndarray) -> np.ndarray:
    initial = np.full(design.shape[1], 1.0 / design.shape[1])

    def objective(weights: np.ndarray) -> float:
        residual = design @ weights - observed
        return 0.5 * float(residual @ residual)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return design.T @ (design @ weights - observed)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * design.shape[1],
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"simplex least-squares fit failed: {result.message}")
    return result.x


def feasible_bounds(
    design: np.ndarray, observed: np.ndarray, tolerance: float, functional: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    upper = np.vstack((design, -design))
    limits = np.concatenate((observed + tolerance, -observed + tolerance))
    equality = np.ones((1, design.shape[1]))
    options = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}
    minimum = linprog(functional, A_ub=upper, b_ub=limits, A_eq=equality,
                      b_eq=np.array([1.0]), bounds=(0.0, None),
                      method="highs", options=options)
    maximum = linprog(-functional, A_ub=upper, b_ub=limits, A_eq=equality,
                      b_eq=np.array([1.0]), bounds=(0.0, None),
                      method="highs", options=options)
    if not minimum.success or not maximum.success:
        raise RuntimeError(f"identified-set LP failed: {minimum.message}; {maximum.message}")
    for witness in (minimum.x, maximum.x):
        if np.max(np.abs(design @ witness - observed)) > tolerance + 2e-7:
            raise AssertionError("LP witness violates response tolerance")
        if abs(witness.sum() - 1.0) > 1e-7 or witness.min() < -1e-7:
            raise AssertionError("LP witness violates spectral constraint")
    return (float(functional @ minimum.x), float(functional @ maximum.x),
            minimum.x, maximum.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[1] /
                        "results" / "p4_kernel_observability_gate_20261010.json")
    args = parser.parse_args()

    dt = 0.1
    times = np.arange(301) * dt
    mids = times[:-1] + dt / 2.0
    rates = np.geomspace(0.02, 4.0, 12)
    true_weights = np.zeros(12)
    true_weights[3], true_weights[9] = 0.7, 0.3
    step = np.ones_like(mids)
    pulses = (((mids % 2.5) < 0.65) | ((mids > 3.2) & (mids < 4.8))).astype(float)
    new_input = 0.2 + 0.8 * np.sin(0.7 * mids) ** 2
    step_design = design_matrix(step, rates, dt)
    pulse_design = design_matrix(pulses, rates, dt)
    new_design = design_matrix(new_input, rates, dt)
    # An independent closed form checks the exact recurrence for step forcing.
    if not np.allclose(step_design, 1.0 - np.exp(-times[:, None] * rates[None, :]),
                       rtol=1e-12, atol=1e-12):
        raise AssertionError("piecewise-constant modal recurrence failed step reference")
    observed_end = 8.0
    last_index = int(round(observed_end / dt))
    single = step_design[np.arange(1, last_index + 1)]
    dual = np.vstack((step_design[np.arange(2, last_index + 1, 2)],
                      pulse_design[np.arange(2, last_index + 1, 2)]))
    assert single.shape == dual.shape == (80, 12)

    target_time = 20.0
    kernel_functional = rates * np.exp(-rates * target_time)
    response_functional = new_design[-1]
    true_kernel = float(kernel_functional @ true_weights)
    true_new_response = float(response_functional @ true_weights)
    noise_bound = 0.02 * float((single @ true_weights).max())
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(20261010 + seed)
        for name, design in (("single_step", single), ("step_and_pulse", dual)):
            clean = design @ true_weights
            observed = clean + rng.uniform(-noise_bound, noise_bound, clean.size)
            assert np.max(np.abs(observed - clean)) <= noise_bound + 1e-12
            estimate = fit_simplex(design, observed)
            klo, khi, low_w, high_w = feasible_bounds(
                design, observed, noise_bound, kernel_functional)
            ylo, yhi, _, _ = feasible_bounds(
                design, observed, noise_bound, response_functional)
            if not (klo - 1e-8 <= true_kernel <= khi + 1e-8):
                raise AssertionError("true kernel excluded despite bounded noise")
            if not (ylo - 1e-8 <= true_new_response <= yhi + 1e-8):
                raise AssertionError("true response excluded despite bounded noise")
            rows.append({
                "seed": seed, "design": name,
                "fit_train_rmse": float(np.sqrt(np.mean((design @ estimate - clean) ** 2))),
                "fit_kernel_at_20": float(kernel_functional @ estimate),
                "true_kernel_at_20": true_kernel,
                "kernel_at_20_interval": [klo, khi],
                "kernel_at_20_interval_width": khi - klo,
                "fit_new_input_response_at_30": float(response_functional @ estimate),
                "true_new_input_response_at_30": true_new_response,
                "new_input_response_at_30_interval": [ylo, yhi],
                "new_input_response_at_30_interval_width": yhi - ylo,
                "extreme_kernel_witnesses": [low_w.tolist(), high_w.tolist()],
            })
    summaries = {}
    for name in ("single_step", "step_and_pulse"):
        group = [row for row in rows if row["design"] == name]
        summaries[name] = {
            key: float(np.mean([row[key] for row in group]))
            for key in ("fit_train_rmse", "kernel_at_20_interval_width",
                        "new_input_response_at_30_interval_width")
        }
    output = {
        "purpose": "Response-only finite-basis identifiability gate; no method novelty claim",
        "assumptions": ["known fixed 12-rate grid", "nonnegative unit-sum weights",
                        "piecewise-constant known inputs", "exact modal recurrence",
                        "independent bounded uniform observation noise with known bound"],
        "rates": rates.tolist(), "true_weights": true_weights.tolist(),
        "dt": dt, "observed_end": observed_end, "measurements_per_design": 80,
        "noise_bound": noise_bound, "seeds": 10,
        "heldout_functionals": ["kernel at time 20", "new-input response at time 30"],
        "summary": summaries, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "noise_bound": noise_bound}, indent=2))


if __name__ == "__main__":
    main()

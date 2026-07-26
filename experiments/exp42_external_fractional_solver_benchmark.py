"""External benchmark on scalar Caputo relaxation.

The comparison is deliberately scoped: dfsc exploits the known analytical
Mittag-Leffler propagator, while FDEint and pycaputo are general history-aware
time steppers.  Results therefore quantify a specialized use advantage, not a
claim that one package replaces the others.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from FDEint import FDEint
from pycaputo.controller import make_fixed_controller
from pycaputo.derivatives import CaputoDerivative
from pycaputo.events import StepCompleted
from pycaputo.fode import caputo
from pycaputo.stepping import evolve
from pymittagleffler import mittag_leffler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dfsc


ALPHAS = (0.55, 0.75, 0.95)
STEP_COUNTS = (64, 256, 1024)
REPEATS = 3
RATE = 1.0
FINAL_TIME = 2.0
TABLES = ROOT / "generated_results"
COUPLED_MATRIX = np.array([[-2.0, 0.6], [0.6, -1.0]], dtype=np.float64)
COUPLED_INITIAL = np.array([1.0, -0.25], dtype=np.float64)


def reference(times: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray([mittag_leffler(-RATE * t**alpha, alpha, 1.0).real for t in times])


def relative_error(values: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(values - target) / np.linalg.norm(target))


def run_dfsc(times: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, object]]:
    time_tensor = torch.as_tensor(times, dtype=torch.float64)
    z = -RATE * time_tensor.pow(alpha)
    result = dfsc.evaluate_mittag_leffler_adaptive(
        torch.tensor(alpha, dtype=torch.float64),
        z,
        method="hybrid",
        term_schedule=(24, 48, 80, 120, 180),
        rtol=1e-10,
        atol=1e-12,
    )
    return result.values.detach().cpu().numpy(), result.diagnostics()


def run_fdeint(times: np.ndarray, alpha: float) -> np.ndarray:
    t = torch.as_tensor(times, dtype=torch.float64)
    step = torch.tensor(FINAL_TIME / (len(times) - 1), dtype=torch.float64)
    values = FDEint(
        lambda current_time, state: -RATE * state,
        t,
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor(alpha, dtype=torch.float64),
        h=step,
        dtype=torch.float64,
    )
    return values[0, :, 0].detach().cpu().numpy()


def run_pycaputo(times: np.ndarray, alpha: float) -> np.ndarray:
    dt = FINAL_TIME / (len(times) - 1)
    method = caputo.PECE(
        ds=(CaputoDerivative(alpha),),
        control=make_fixed_controller(dt, tstart=0.0, tfinal=FINAL_TIME),
        source=lambda current_time, state: -RATE * state,
        y0=(np.array([1.0]),),
        corrector_iterations=1,
    )
    events = [event for event in evolve(method, dtinit=dt) if isinstance(event, StepCompleted)]
    event_times = np.asarray([float(event.t) for event in events])
    event_values = np.asarray([float(event.y[0]) for event in events])
    return np.interp(times, event_times, event_values)


def coupled_reference(times: np.ndarray, alpha: float) -> np.ndarray:
    rates, vectors = np.linalg.eigh(-COUPLED_MATRIX)
    modal_initial = vectors.T @ COUPLED_INITIAL
    multipliers = np.asarray(
        [[mittag_leffler(-rate * t**alpha, alpha, 1.0).real for rate in rates] for t in times]
    )
    return (multipliers * modal_initial[None, :]) @ vectors.T


def run_dfsc_coupled(times: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, object]]:
    rates, vectors = torch.linalg.eigh(torch.as_tensor(-COUPLED_MATRIX, dtype=torch.float64))
    initial = torch.as_tensor(COUPLED_INITIAL, dtype=torch.float64)
    modal_initial = vectors.T @ initial
    time_tensor = torch.as_tensor(times, dtype=torch.float64)
    evaluation = dfsc.evaluate_mittag_leffler_adaptive(
        torch.tensor(alpha, dtype=torch.float64),
        -time_tensor[:, None].pow(alpha) * rates[None, :],
        method="hybrid",
        term_schedule=(24, 48, 80, 120, 180),
        rtol=1e-10,
        atol=1e-12,
    )
    values = (evaluation.values * modal_initial[None, :]) @ vectors.T
    return values.detach().cpu().numpy(), evaluation.diagnostics()


def run_fdeint_coupled(times: np.ndarray, alpha: float) -> np.ndarray:
    t = torch.as_tensor(times, dtype=torch.float64)
    matrix = torch.as_tensor(COUPLED_MATRIX, dtype=torch.float64)
    values = FDEint(
        lambda current_time, state: state @ matrix.T,
        t,
        torch.as_tensor(COUPLED_INITIAL[None, :], dtype=torch.float64),
        torch.tensor(alpha, dtype=torch.float64),
        h=torch.tensor(FINAL_TIME / (len(times) - 1), dtype=torch.float64),
        dtype=torch.float64,
    )
    return values[0].detach().cpu().numpy()


def run_pycaputo_coupled(times: np.ndarray, alpha: float) -> np.ndarray:
    dt = FINAL_TIME / (len(times) - 1)
    method = caputo.PECE(
        ds=(CaputoDerivative(alpha), CaputoDerivative(alpha)),
        control=make_fixed_controller(dt, tstart=0.0, tfinal=FINAL_TIME),
        source=lambda current_time, state: COUPLED_MATRIX @ state,
        y0=(COUPLED_INITIAL.copy(),),
        corrector_iterations=1,
    )
    events = [event for event in evolve(method, dtinit=dt) if isinstance(event, StepCompleted)]
    event_times = np.asarray([float(event.t) for event in events])
    event_values = np.asarray([event.y for event in events])
    return np.column_stack(
        [np.interp(times, event_times, event_values[:, component]) for component in range(2)]
    )


def timed(callable_object) -> tuple[np.ndarray, float, dict[str, object]]:
    durations: list[float] = []
    result: np.ndarray | None = None
    diagnostics: dict[str, object] = {}
    for _ in range(REPEATS):
        started = time.perf_counter()
        output = callable_object()
        durations.append(time.perf_counter() - started)
        if isinstance(output, tuple):
            result, diagnostics = output
        else:
            result = output
    assert result is not None
    return result, float(np.median(durations)), diagnostics


def gradient_audit() -> dict[str, object]:
    alpha_dfsc = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    t_dfsc = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    dfsc_values = dfsc.evaluate_mittag_leffler_adaptive(
        alpha_dfsc, -t_dfsc.pow(alpha_dfsc), method="hybrid", rtol=1e-9
    ).values
    dfsc_values[-1].backward()

    alpha_fdeint = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
    t_fdeint = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    fdeint_values = FDEint(
        lambda current_time, state: -state,
        t_fdeint,
        torch.tensor([[1.0]], dtype=torch.float64),
        alpha_fdeint,
        h=torch.tensor(1.0 / 64, dtype=torch.float64),
        dtype=torch.float64,
    )
    fdeint_values[0, -1, 0].backward()
    return {
        "dfsc_alpha_gradient_finite": bool(torch.isfinite(alpha_dfsc.grad).item()),
        "dfsc_alpha_gradient": float(alpha_dfsc.grad),
        "fdeint_alpha_gradient_finite": bool(torch.isfinite(alpha_fdeint.grad).item()),
        "fdeint_alpha_gradient": float(alpha_fdeint.grad),
        "pycaputo_native_autograd": False,
        "interpretation": "FDEint and dfsc both expose alpha gradients; pycaputo uses NumPy stepping.",
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for alpha in ALPHAS:
        for steps in STEP_COUNTS:
            times = np.linspace(0.0, FINAL_TIME, steps + 1)
            target = reference(times, alpha)
            methods = {
                "dfsc adaptive direct query": lambda: run_dfsc(times, alpha),
                "FDEint predictor-corrector": lambda: run_fdeint(times, alpha),
                "pycaputo PECE": lambda: run_pycaputo(times, alpha),
            }
            for method, callable_object in methods.items():
                values, elapsed, diagnostics = timed(callable_object)
                rows.append(
                    {
                        "task": "scalar relaxation",
                        "method": method,
                        "alpha": alpha,
                        "time_points": steps + 1,
                        "relative_l2_error": relative_error(values, target),
                        "max_absolute_error": float(np.max(np.abs(values - target))),
                        "median_seconds": elapsed,
                        "selected_terms": diagnostics.get("selected_terms"),
                        "adaptive_converged": diagnostics.get("adaptive_converged"),
                    }
                )

    for alpha in ALPHAS:
        for steps in (256, 1024):
            times = np.linspace(0.0, FINAL_TIME, steps + 1)
            target = coupled_reference(times, alpha)
            methods = {
                "dfsc adaptive direct query": lambda: run_dfsc_coupled(times, alpha),
                "FDEint predictor-corrector": lambda: run_fdeint_coupled(times, alpha),
                "pycaputo PECE": lambda: run_pycaputo_coupled(times, alpha),
            }
            for method, callable_object in methods.items():
                values, elapsed, diagnostics = timed(callable_object)
                rows.append(
                    {
                        "task": "coupled 2-state relaxation",
                        "method": method,
                        "alpha": alpha,
                        "time_points": steps + 1,
                        "relative_l2_error": relative_error(values, target),
                        "max_absolute_error": float(np.max(np.abs(values - target))),
                        "median_seconds": elapsed,
                        "selected_terms": diagnostics.get("selected_terms"),
                        "adaptive_converged": diagnostics.get("adaptive_converged"),
                    }
                )

    TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES / "external_fractional_solver_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "benchmark": "scalar and coupled two-state linear Caputo relaxation",
        "scope_boundary": "dfsc uses the known propagator; FDEint and pycaputo are general time steppers",
        "hardware": {
            "torch_device": "cpu",
            "torch_version": torch.__version__,
        },
        "versions": {
            "dfsc": dfsc.__version__,
            "FDEint": importlib.metadata.version("FDEint"),
            "pycaputo": importlib.metadata.version("pycaputo"),
            "pymittagleffler_reference": importlib.metadata.version("pymittagleffler"),
        },
        "gradient_audit": gradient_audit(),
        "rows": rows,
    }
    output_path = TABLES / "external_fractional_solver_benchmark_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

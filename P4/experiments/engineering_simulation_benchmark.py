"""Engineering-simulation benchmark for the P4 reliability protocol.

The benchmark keeps the dynamical system fixed and varies only the numerical
propagation rule.  It measures trajectory error at several horizons and RK4
step sizes against the exact matrix exponential solution.  This is a
simulation-oriented validation of the protocol, not a claim that RK4 or the
matrix action is universally optimal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
from experiments.p4_generic_matrix_exp_validation import MatrixExponentialAction  # noqa: E402
from experiments.p4_generic_ode_step_validation import RK4LinearStep  # noqa: E402


def exact_trajectory(matrix: torch.Tensor, state: torch.Tensor, time: float) -> torch.Tensor:
    return torch.matrix_exp(time * matrix) @ state


def rk4_trajectory(stepper, matrix: torch.Tensor, state: torch.Tensor, time: float, step: float) -> torch.Tensor:
    n_steps = int(round(time / step))
    current = state
    for _ in range(n_steps):
        inputs = torch.cat((torch.tensor([step], dtype=state.dtype), current)).unsqueeze(0)
        current = stepper(inputs, matrix).squeeze(0)
    return current


def main() -> None:
    torch.set_default_dtype(torch.float64)
    matrix = torch.tensor([[-0.70, 0.20], [-0.10, -0.40]])
    horizons = [1.0, 2.0, 4.0, 8.0, 12.0]
    steps = [0.20, 0.10, 0.05]
    seeds = [901, 902, 903, 904, 905]
    matrix_action = MatrixExponentialAction()
    rk4 = RK4LinearStep()
    rows = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        state = torch.randn(2, generator=generator)
        for horizon in horizons:
            reference = exact_trajectory(matrix, state, horizon)
            exact_action = matrix_action(
                torch.cat((torch.tensor([horizon]), state)).unsqueeze(0), matrix
            ).squeeze(0)
            rows.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "matrix_action_abs_error": float((exact_action - reference).abs().max()),
                    "rk4": {
                        str(step): float(
                            (rk4_trajectory(rk4, matrix, state, horizon, step) - reference).abs().max()
                        )
                        for step in steps
                    },
                }
            )
    result = {
        "schema": "DFSC-P4-Engineering-Simulation-Benchmark-v1",
        "system": "y_prime = A y",
        "matrix": matrix.tolist(),
        "horizons": horizons,
        "rk4_steps": steps,
        "seeds": seeds,
        "reference": "torch.matrix_exp applied to the same stable 2x2 system",
        "rows": rows,
        "interpretation": (
            "Trajectory-error evidence for simulation-oriented reliability; "
            "not a universal solver ranking or a cross-hardware benchmark."
        ),
    }
    out = ROOT / "P4" / "results" / "p4_engineering_simulation_benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

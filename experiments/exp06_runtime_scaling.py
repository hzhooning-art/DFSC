"""Runtime and history-storage scaling against an L1 time-marching baseline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import l1_caputo_relaxation, mittag_leffler_e


def time_call(fn, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return min(timings)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    alpha = torch.tensor(0.65)
    mu = torch.tensor(3.0)
    final_time = 0.20
    u0 = torch.tensor(1.0)

    print("N_t,mlsl_seconds,l1_seconds,estimated_l1_history_kb,estimated_mlsl_state_kb")
    for num_steps in [50, 100, 200, 400, 800]:
        times = torch.linspace(0.0, final_time, num_steps + 1)

        def run_mlsl() -> torch.Tensor:
            z = -mu * times.pow(alpha)
            return u0 * mittag_leffler_e(alpha, z, terms=80)

        def run_l1() -> torch.Tensor:
            return l1_caputo_relaxation(
                u0,
                alpha=alpha,
                mu=mu,
                final_time=final_time,
                num_steps=num_steps,
            )

        mlsl_seconds = time_call(run_mlsl)
        l1_seconds = time_call(run_l1)

        # These are simple evidence-oriented estimates for the scalar history.
        # In a full PDE solver, multiply by the number of spatial states/modes.
        estimated_l1_history_kb = (num_steps + 1) * 8 / 1024
        estimated_mlsl_state_kb = 3 * 8 / 1024
        print(
            f"{num_steps},{mlsl_seconds:.6e},{l1_seconds:.6e},"
            f"{estimated_l1_history_kb:.6f},{estimated_mlsl_state_kb:.6f}"
        )


if __name__ == "__main__":
    main()

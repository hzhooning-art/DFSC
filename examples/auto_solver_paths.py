"""Demonstrate dfsc's direct-query and history-aware solve paths."""

from __future__ import annotations

import torch

import dfsc


def main() -> None:
    torch.set_default_dtype(torch.float64)
    x, layer = dfsc.build_mlsl(
        dimension=1,
        boundary="dirichlet",
        num_points=48,
        num_modes=10,
    )
    spectral_problem = dfsc.FractionalSpectralProblem(
        layer=layer,
        u0=torch.sin(torch.pi * x),
        times=torch.linspace(0.0, 0.08, 6),
        alpha=torch.tensor(0.75, requires_grad=True),
    )
    spectral_solution = dfsc.solve(spectral_problem)

    history_problem = dfsc.CaputoL1Problem(
        operator=torch.tensor([[1.0, -0.2], [-0.2, 1.5]]),
        u0=torch.tensor([1.0, 0.0]),
        alpha=torch.tensor(0.75, requires_grad=True),
        final_time=0.08,
        num_steps=20,
    )
    history_solution = dfsc.solve(history_problem)

    print("spectral:", spectral_solution.algorithm, spectral_solution.diagnostics)
    print("history:", history_solution.algorithm, history_solution.stats)


if __name__ == "__main__":
    main()

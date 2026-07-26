"""Minimal problem--algorithm--solve example for dfsc."""

from __future__ import annotations

import torch

import dfsc


def main() -> None:
    torch.set_default_dtype(torch.float64)
    x, layer = dfsc.build_mlsl(
        dimension=1,
        boundary="dirichlet",
        num_points=64,
        num_modes=12,
        config=dfsc.MLSLConfig.stable(terms=80),
    )
    problem = dfsc.FractionalSpectralProblem(
        layer=layer,
        u0=torch.sin(torch.pi * x),
        times=torch.linspace(0.0, 0.05, 8),
        alpha=torch.tensor(0.9, requires_grad=True),
        beta=torch.tensor(1.4, requires_grad=True),
        metadata={"example": "problem-solve-interface"},
    )
    solution = dfsc.solve(problem, dfsc.MLSLStable(terms=80))
    loss = solution.final.square().mean()
    loss.backward()
    print(
        {
            "algorithm": solution.algorithm,
            "shape": tuple(solution.values.shape),
            "finite": bool(torch.isfinite(solution.values).all()),
            "alpha_grad": float(problem.alpha.grad),
            "beta_grad": float(problem.beta.grad),
        }
    )


if __name__ == "__main__":
    main()

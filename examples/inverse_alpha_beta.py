"""Recover alpha and beta from synthetic observations with MLSL."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, build_dirichlet_mlsl_1d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def main() -> None:
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    x, layer = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=14,
        config=MLSLConfig(terms=90),
    )
    u0 = torch.sin(torch.pi * x) + 0.25 * torch.sin(2.0 * torch.pi * x)
    times = torch.linspace(0.0, 0.025, 7)
    alpha_true = torch.tensor(1.38)
    beta_true = torch.tensor(1.45)
    observed = layer(u0, times, alpha_true, beta=beta_true).detach()

    raw_alpha = torch.nn.Parameter(torch.tensor(0.7))
    raw_beta = torch.nn.Parameter(torch.tensor(0.2))
    opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.05)

    for _ in range(250):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        loss = torch.mean((layer(u0, times, alpha, beta=beta) - observed) ** 2)
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).detach()
    beta_est = constrain(raw_beta, 0.60, 1.95).detach()
    print("alpha_true:", float(alpha_true), "alpha_est:", float(alpha_est))
    print("beta_true:", float(beta_true), "beta_est:", float(beta_est))
    print("final_loss:", float(loss.detach()))


if __name__ == "__main__":
    main()

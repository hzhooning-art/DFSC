"""Jointly recover hidden alpha and beta from synthetic observations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(11)

    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=18)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)

    u0 = (
        torch.sin(torch.pi * x)
        + 0.25 * torch.sin(2.0 * torch.pi * x)
        + 0.12 * torch.sin(5.0 * torch.pi * x)
    )
    times = torch.linspace(0.0, 0.025, 8)
    alpha_true = torch.tensor(1.38)
    beta_true = torch.tensor(1.35)
    observations = layer(u0, times, alpha_true, beta=beta_true).detach()
    observations = observations + 5e-5 * torch.randn_like(observations)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
    raw_beta = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)

    for step in range(501):
        optimizer.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        pred = layer(u0, times, alpha, beta=beta)
        loss = torch.mean((pred - observations) ** 2)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 500:
            print(
                f"step={step:04d} loss={loss.item():.6e} "
                f"alpha={alpha.item():.6f} beta={beta.item():.6f}"
            )

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est)
    print("alpha_relative_error:", abs(alpha_est - alpha_true.item()) / alpha_true.item())
    print("beta_true:", beta_true.item())
    print("beta_est:", beta_est)
    print("beta_relative_error:", abs(beta_est - beta_true.item()) / beta_true.item())


if __name__ == "__main__":
    main()

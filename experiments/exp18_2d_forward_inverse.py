"""Small 2D tensor-product spectral example for MLSL."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_2d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(71)

    coords, eigenvalues, phi = dirichlet_laplacian_2d(num_points_1d=18, num_modes_1d=6)
    x = coords[:, 0]
    y = coords[:, 1]
    u0 = (
        torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        + 0.2 * torch.sin(2.0 * torch.pi * x) * torch.sin(3.0 * torch.pi * y)
    )
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    times = torch.linspace(0.0, 0.025, 7)
    alpha_true = torch.tensor(1.36)
    beta_true = torch.tensor(1.55)
    observations = layer(u0, times, alpha_true, beta=beta_true).detach()
    observations = observations + 1e-4 * torch.std(observations) * torch.randn_like(observations)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.8))
    raw_beta = torch.nn.Parameter(torch.tensor(0.4))
    opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
    for _ in range(450):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        loss = torch.mean((layer(u0, times, alpha, beta=beta) - observations) ** 2)
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    print("num_points:", coords.shape[0])
    print("num_modes:", eigenvalues.numel())
    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est)
    print("alpha_rel_error:", abs(alpha_est - alpha_true.item()) / alpha_true.item())
    print("beta_true:", beta_true.item())
    print("beta_est:", beta_est)
    print("beta_rel_error:", abs(beta_est - beta_true.item()) / beta_true.item())
    print("final_loss:", loss.item())


if __name__ == "__main__":
    main()

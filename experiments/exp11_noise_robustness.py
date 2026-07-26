"""Noise robustness for hidden alpha/beta recovery."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def run_one(noise_level: float) -> tuple[float, float, float]:
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
    clean = layer(u0, times, alpha_true, beta=beta_true).detach()
    scale = torch.std(clean)
    observed = clean + noise_level * scale * torch.randn_like(clean)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
    raw_beta = torch.nn.Parameter(torch.tensor(0.5))
    opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
    for _ in range(500):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        loss = torch.mean((layer(u0, times, alpha, beta=beta) - observed) ** 2)
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    rel_alpha = abs(alpha_est - alpha_true.item()) / alpha_true.item()
    rel_beta = abs(beta_est - beta_true.item()) / beta_true.item()
    return rel_alpha, rel_beta, loss.item()


def main() -> None:
    torch.set_default_dtype(torch.float64)
    print("noise_level,alpha_rel_error,beta_rel_error,final_loss")
    for idx, noise in enumerate([0.0, 1e-4, 1e-3, 1e-2, 5e-2]):
        torch.manual_seed(100 + idx)
        alpha_err, beta_err, loss = run_one(noise)
        print(f"{noise:.1e},{alpha_err:.6e},{beta_err:.6e},{loss:.6e}")


if __name__ == "__main__":
    main()

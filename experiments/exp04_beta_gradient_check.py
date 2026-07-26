"""Finite-difference check for gradients through beta and modal rates."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def loss_at_beta(layer: MittagLefflerSpectralLayer, u0: torch.Tensor, target: torch.Tensor, beta_value: float) -> torch.Tensor:
    alpha = torch.tensor(1.35, dtype=u0.dtype)
    beta = torch.tensor(beta_value, dtype=u0.dtype)
    pred = layer(u0, torch.tensor(0.015, dtype=u0.dtype), alpha, beta=beta)
    return torch.mean((pred - target) ** 2)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=96, num_modes=14)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)

    u0 = torch.sin(torch.pi * x) + 0.20 * torch.sin(3.0 * torch.pi * x)
    alpha_true = torch.tensor(1.35)
    beta_true = torch.tensor(1.25)
    target = layer(u0, torch.tensor(0.015), alpha_true, beta=beta_true).detach()

    beta = torch.tensor(1.65, requires_grad=True)
    pred = layer(u0, torch.tensor(0.015), torch.tensor(1.35), beta=beta)
    loss = torch.mean((pred - target) ** 2)
    loss.backward()
    grad_auto = beta.grad.detach().item()

    print("autograd beta grad:", grad_auto)
    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        lp = loss_at_beta(layer, u0, target, 1.65 + eps)
        lm = loss_at_beta(layer, u0, target, 1.65 - eps)
        grad_fd = ((lp - lm) / (2.0 * eps)).item()
        rel = abs(grad_auto - grad_fd) / max(abs(grad_fd), 1e-14)
        print(f"eps={eps:.1e} grad_fd={grad_fd:.8e} rel_error={rel:.3e}")


if __name__ == "__main__":
    main()

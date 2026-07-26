"""Finite-difference check for gradients through alpha."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def loss_at_alpha(layer: MittagLefflerSpectralLayer, u0: torch.Tensor, target: torch.Tensor, alpha_value: float) -> torch.Tensor:
    alpha = torch.tensor(alpha_value, dtype=u0.dtype)
    pred = layer(u0, torch.tensor(0.05, dtype=u0.dtype), alpha)
    return torch.mean((pred - target) ** 2)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=96, num_modes=16)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=90)

    u0 = torch.sin(torch.pi * x) + 0.15 * torch.sin(2.0 * torch.pi * x)
    alpha_true = torch.tensor(1.35)
    target = layer(u0, torch.tensor(0.05), alpha_true).detach()

    alpha = torch.tensor(1.65, requires_grad=True)
    pred = layer(u0, torch.tensor(0.05), alpha)
    loss = torch.mean((pred - target) ** 2)
    loss.backward()
    grad_auto = alpha.grad.detach().item()

    print("autograd grad:", grad_auto)
    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        lp = loss_at_alpha(layer, u0, target, 1.65 + eps)
        lm = loss_at_alpha(layer, u0, target, 1.65 - eps)
        grad_fd = ((lp - lm) / (2.0 * eps)).item()
        rel = abs(grad_auto - grad_fd) / max(abs(grad_fd), 1e-14)
        print(f"eps={eps:.1e} grad_fd={grad_fd:.8e} rel_error={rel:.3e}")


if __name__ == "__main__":
    main()

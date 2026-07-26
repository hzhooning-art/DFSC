"""Recover a hidden fractional order alpha from synthetic observations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def constrained_alpha(raw_alpha: torch.Tensor, low: float = 1.05, high: float = 1.95) -> torch.Tensor:
    """Map an unconstrained parameter to a stable alpha interval."""

    return low + (high - low) * torch.sigmoid(raw_alpha)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(7)

    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=20)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=90)

    u0 = torch.sin(torch.pi * x) + 0.20 * torch.sin(4.0 * torch.pi * x)
    times = torch.linspace(0.0, 0.08, 9)
    alpha_true = torch.tensor(1.42)
    observations = layer(u0, times, alpha_true).detach()

    # Add a tiny amount of noise so the inverse task is still realistic.
    observations = observations + 1e-4 * torch.randn_like(observations)

    raw_alpha = torch.nn.Parameter(torch.tensor(1.2))
    optimizer = torch.optim.Adam([raw_alpha], lr=0.05)

    for step in range(401):
        optimizer.zero_grad()
        alpha = constrained_alpha(raw_alpha)
        pred = layer(u0, times, alpha)
        loss = torch.mean((pred - observations) ** 2)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 400:
            print(f"step={step:04d} loss={loss.item():.6e} alpha={alpha.item():.6f}")

    alpha_est = constrained_alpha(raw_alpha).item()
    rel_error = abs(alpha_est - alpha_true.item()) / alpha_true.item()
    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est)
    print("relative_error:", rel_error)


if __name__ == "__main__":
    main()

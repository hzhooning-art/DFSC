"""Forward demo for the first-stage Mittag-Leffler spectral layer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def main() -> None:
    torch.set_default_dtype(torch.float64)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=24)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=80)

    u0 = torch.sin(torch.pi * x) + 0.25 * torch.sin(3.0 * torch.pi * x)
    times = torch.linspace(0.0, 0.25, 6)
    alpha = torch.tensor(1.6)
    u = layer(u0, times, alpha)

    print("x shape:", tuple(x.shape))
    print("u shape:", tuple(u.shape))
    print("times:", times.tolist())
    print("u(t=0) relative reconstruction error:", torch.linalg.norm(u[0] - u0).item() / torch.linalg.norm(u0).item())
    print("u last min/max:", float(u[-1].min()), float(u[-1].max()))


if __name__ == "__main__":
    main()

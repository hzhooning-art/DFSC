"""Minimal forward use of the Mittag-Leffler spectral layer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, build_dirichlet_mlsl_1d


def main() -> None:
    torch.set_default_dtype(torch.float64)
    x, layer = build_dirichlet_mlsl_1d(
        num_points=64,
        num_modes=12,
        config=MLSLConfig(terms=90),
    )
    u0 = torch.sin(torch.pi * x) + 0.2 * torch.sin(3.0 * torch.pi * x)
    times = torch.linspace(0.0, 0.04, 6)
    alpha = torch.tensor(1.35, requires_grad=True)

    u = layer(u0, times, alpha)
    loss = torch.mean(u[-1] ** 2)
    loss.backward()

    print("output_shape:", tuple(u.shape))
    print("loss:", float(loss.detach()))
    print("d_loss_d_alpha:", float(alpha.grad.detach()))


if __name__ == "__main__":
    main()

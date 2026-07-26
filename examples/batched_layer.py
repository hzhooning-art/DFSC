"""Batched MLSL calls for dataset-level SciML workflows."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, build_dirichlet_mlsl_1d


def main() -> None:
    torch.manual_seed(5)
    torch.set_default_dtype(torch.float64)
    x, layer = build_dirichlet_mlsl_1d(
        num_points=80,
        num_modes=16,
        config=MLSLConfig(terms=90),
    )
    coeffs = torch.randn(8, 4)
    modes = torch.stack([torch.sin((i + 1) * torch.pi * x) for i in range(4)])
    u0_batch = coeffs @ modes
    times = torch.linspace(0.0, 0.035, 5)
    out = layer(u0_batch, times, torch.tensor(1.42), beta=torch.tensor(1.30))
    print("u0_batch_shape:", tuple(u0_batch.shape))
    print("output_shape:", tuple(out.shape))


if __name__ == "__main__":
    main()

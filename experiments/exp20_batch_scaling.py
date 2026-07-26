"""Batch evaluation scaling for the MLSL primitive."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d
from experiments.exp14_fno_dataset_long_time import make_random_initial_conditions


def timed(fn, repeats: int = 3) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return min(values)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(91)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=24)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    times = torch.linspace(0.0, 0.04, 8)
    alpha = torch.tensor(1.42)

    print("batch_size,seconds,output_shape")
    for batch_size in [1, 4, 16, 64, 128]:
        u0 = make_random_initial_conditions(x, count=batch_size)

        def run() -> torch.Tensor:
            return layer(u0, times, alpha)

        output = run()
        seconds = timed(run)
        print(f"{batch_size},{seconds:.6e},{tuple(output.shape)}")


if __name__ == "__main__":
    main()

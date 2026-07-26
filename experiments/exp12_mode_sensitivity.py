"""Mode truncation sensitivity for forward accuracy and alpha recovery."""

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
    torch.manual_seed(31)

    num_points = 128
    x_ref, eigenvalues_ref, phi_ref = dirichlet_laplacian_1d(num_points=num_points, num_modes=64)
    ref_layer = MittagLefflerSpectralLayer(eigenvalues_ref, phi_ref, terms=100)
    u0 = torch.sin(torch.pi * x_ref) + 0.20 * torch.sin(6.0 * torch.pi * x_ref)
    times = torch.linspace(0.0, 0.035, 8)
    alpha_true = torch.tensor(1.42)
    reference = ref_layer(u0, times, alpha_true).detach()

    print("num_modes,forward_rel_error,alpha_est,alpha_rel_error")
    for num_modes in [4, 8, 12, 16, 24, 32, 48]:
        _, eigenvalues, phi = dirichlet_laplacian_1d(num_points=num_points, num_modes=num_modes)
        layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
        forward = layer(u0, times, alpha_true).detach()
        forward_error = (torch.linalg.norm(forward - reference) / torch.linalg.norm(reference)).item()

        raw_alpha = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.Adam([raw_alpha], lr=0.05)
        for _ in range(250):
            opt.zero_grad()
            alpha = constrain(raw_alpha, 1.05, 1.95)
            loss = torch.mean((layer(u0, times, alpha) - reference) ** 2)
            loss.backward()
            opt.step()

        alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
        alpha_err = abs(alpha_est - alpha_true.item()) / alpha_true.item()
        print(f"{num_modes},{forward_error:.6e},{alpha_est:.8f},{alpha_err:.6e}")


if __name__ == "__main__":
    main()

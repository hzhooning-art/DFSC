"""Recover alpha/beta from sparse sensors and sparse query times."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def run_case(num_sensors: int, num_times: int) -> tuple[float, float, float]:
    torch.manual_seed(200 + num_sensors + num_times)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=18)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    u0 = (
        torch.sin(torch.pi * x)
        + 0.25 * torch.sin(2.0 * torch.pi * x)
        + 0.12 * torch.sin(5.0 * torch.pi * x)
    )
    full_times = torch.linspace(0.0, 0.03, 10)
    time_idx = torch.linspace(0, full_times.numel() - 1, num_times).round().long()
    sensor_idx = torch.linspace(4, x.numel() - 5, num_sensors).round().long()
    times = full_times[time_idx]

    alpha_true = torch.tensor(1.38)
    beta_true = torch.tensor(1.35)
    clean = layer(u0, times, alpha_true, beta=beta_true).detach()
    observed = clean[:, sensor_idx] + 1e-4 * torch.std(clean) * torch.randn(num_times, num_sensors)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
    raw_beta = torch.nn.Parameter(torch.tensor(0.5))
    opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
    for _ in range(600):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        pred = layer(u0, times, alpha, beta=beta)[:, sensor_idx]
        loss = torch.mean((pred - observed) ** 2)
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    return (
        abs(alpha_est - alpha_true.item()) / alpha_true.item(),
        abs(beta_est - beta_true.item()) / beta_true.item(),
        loss.item(),
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    print("num_sensors,num_times,alpha_rel_error,beta_rel_error,final_loss")
    for sensors, times in [(4, 4), (8, 4), (8, 6), (16, 6)]:
        alpha_err, beta_err, loss = run_case(sensors, times)
        print(f"{sensors},{times},{alpha_err:.6e},{beta_err:.6e},{loss:.6e}")


if __name__ == "__main__":
    main()

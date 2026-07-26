"""Long-time prediction: MLSL structure vs a black-box MLP neural field."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MLPField, MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(23)

    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=64, num_modes=16)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    u0 = torch.sin(torch.pi * x) + 0.20 * torch.sin(3.0 * torch.pi * x)
    alpha_true = torch.tensor(1.42)

    train_times = torch.linspace(0.0, 0.04, 8)
    test_times = torch.linspace(0.05, 0.16, 12)
    y_train = layer(u0, train_times, alpha_true).detach()
    y_test = layer(u0, test_times, alpha_true).detach()

    raw_alpha = torch.nn.Parameter(torch.tensor(1.0))
    opt_alpha = torch.optim.Adam([raw_alpha], lr=0.05)
    for _ in range(250):
        opt_alpha.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        loss = torch.mean((layer(u0, train_times, alpha) - y_train) ** 2)
        loss.backward()
        opt_alpha.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).detach()
    mlsl_train = layer(u0, train_times, alpha_est).detach()
    mlsl_test = layer(u0, test_times, alpha_est).detach()

    mlp = MLPField(hidden=64, depth=3).to(dtype=torch.float64)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=2e-3)
    x_grid = x[None, :].expand(train_times.numel(), -1)
    t_grid = train_times[:, None].expand(-1, x.numel())
    for _ in range(800):
        opt_mlp.zero_grad()
        pred = mlp(x_grid, t_grid)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt_mlp.step()

    x_test = x[None, :].expand(test_times.numel(), -1)
    t_test = test_times[:, None].expand(-1, x.numel())
    mlp_train = mlp(x_grid, t_grid).detach()
    mlp_test = mlp(x_test, t_test).detach()

    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est.item())
    print("mlsl_train_rel_error:", relative_error(mlsl_train, y_train))
    print("mlsl_long_time_rel_error:", relative_error(mlsl_test, y_test))
    print("mlp_train_rel_error:", relative_error(mlp_train, y_train))
    print("mlp_long_time_rel_error:", relative_error(mlp_test, y_test))


if __name__ == "__main__":
    main()

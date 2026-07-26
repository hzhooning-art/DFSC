"""Dataset-level long-time operator learning with a DeepONet baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import DeepONet1D, MittagLefflerSpectralLayer, dirichlet_laplacian_1d
from experiments.exp14_fno_dataset_long_time import (
    flatten_dataset,
    make_random_initial_conditions,
    rel,
)


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(51)

    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=64, num_modes=24)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    alpha_true = torch.tensor(1.42)
    train_times = torch.linspace(0.0, 0.04, 6)
    test_times = torch.linspace(0.06, 0.16, 8)
    train_u0 = make_random_initial_conditions(x, count=24)
    test_u0 = make_random_initial_conditions(x, count=8)

    train_u0_rows, train_t_rows, train_y = flatten_dataset(train_u0, train_times, layer, alpha_true)
    test_u0_rows, test_t_rows, test_y = flatten_dataset(test_u0, test_times, layer, alpha_true)
    x_train = x[None, :].expand(train_u0_rows.shape[0], -1)
    x_test = x[None, :].expand(test_u0_rows.shape[0], -1)
    t_train = train_t_rows[:, None].expand(-1, x.numel())
    t_test = test_t_rows[:, None].expand(-1, x.numel())

    raw_alpha = torch.nn.Parameter(torch.tensor(1.0))
    opt_alpha = torch.optim.Adam([raw_alpha], lr=0.05)
    for _ in range(250):
        opt_alpha.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        pred = torch.stack([layer(u0, t, alpha) for u0, t in zip(train_u0_rows, train_t_rows, strict=True)])
        loss = torch.mean((pred - train_y) ** 2)
        loss.backward()
        opt_alpha.step()
    alpha_est = constrain(raw_alpha, 1.05, 1.95).detach()
    _, _, mlsl_test = flatten_dataset(test_u0, test_times, layer, alpha_est)

    deeponet = DeepONet1D(num_points=x.numel(), latent=64, hidden=96).to(dtype=torch.float64)
    opt = torch.optim.Adam(deeponet.parameters(), lr=1e-3)
    alpha_train = alpha_true.expand(train_u0_rows.shape[0])
    alpha_test = alpha_true.expand(test_u0_rows.shape[0])
    y_mean = train_y.mean()
    y_std = train_y.std().clamp_min(1e-12)
    train_y_norm = (train_y - y_mean) / y_std
    for _ in range(700):
        opt.zero_grad()
        pred = deeponet(train_u0_rows, x_train, t_train, alpha_train)
        loss = torch.mean((pred - train_y_norm) ** 2)
        loss.backward()
        opt.step()

    deeponet_train = (deeponet(train_u0_rows, x_train, t_train, alpha_train) * y_std + y_mean).detach()
    deeponet_test = (deeponet(test_u0_rows, x_test, t_test, alpha_test) * y_std + y_mean).detach()

    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est.item())
    print("mlsl_test_long_time_rel_error:", rel(mlsl_test, test_y))
    print("deeponet_train_rel_error:", rel(deeponet_train, train_y))
    print("deeponet_test_long_time_rel_error:", rel(deeponet_test, test_y))


if __name__ == "__main__":
    main()

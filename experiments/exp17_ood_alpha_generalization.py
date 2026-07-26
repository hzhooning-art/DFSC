"""OOD alpha generalization: exact MLSL parameterization vs DeepONet."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import ConditionalMLPField, MittagLefflerSpectralLayer, dirichlet_laplacian_1d
from experiments.exp14_fno_dataset_long_time import make_random_initial_conditions, rel


def build_rows(u0_batch: torch.Tensor, alphas: torch.Tensor, times: torch.Tensor, layer: MittagLefflerSpectralLayer):
    u0_rows, alpha_rows, t_rows, y_rows = [], [], [], []
    for u0 in u0_batch:
        for alpha in alphas:
            y = layer(u0, times, alpha).detach()
            u0_rows.append(u0[None, :].expand(times.numel(), -1))
            alpha_rows.append(alpha.expand(times.numel()))
            t_rows.append(times)
            y_rows.append(y)
    return torch.cat(u0_rows), torch.cat(alpha_rows), torch.cat(t_rows), torch.cat(y_rows)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(61)

    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=64, num_modes=8)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    times = torch.linspace(0.0, 0.02, 8)
    train_alphas = torch.tensor([1.15, 1.25, 1.35, 1.45, 1.55])
    ood_alphas = torch.tensor([1.70, 1.80])
    base_u0 = make_random_initial_conditions(x, count=1)
    train_u0 = base_u0
    test_u0 = base_u0

    train_u0_rows, train_alpha_rows, train_t_rows, train_y = build_rows(train_u0, train_alphas, times, layer)
    test_u0_rows, test_alpha_rows, test_t_rows, test_y = build_rows(test_u0, ood_alphas, times, layer)
    x_train = x[None, :].expand(train_u0_rows.shape[0], -1)
    x_test = x[None, :].expand(test_u0_rows.shape[0], -1)
    t_train = train_t_rows[:, None].expand(-1, x.numel())
    t_test = test_t_rows[:, None].expand(-1, x.numel())

    model = ConditionalMLPField(hidden=128, depth=4).to(dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    y_mean = train_y.mean()
    y_std = train_y.std().clamp_min(1e-12)
    train_y_norm = (train_y - y_mean) / y_std
    for _ in range(4000):
        opt.zero_grad()
        pred = model(x_train, t_train, train_alpha_rows)
        loss = torch.mean((pred - train_y_norm) ** 2)
        loss.backward()
        opt.step()

    deep_train = (model(x_train, t_train, train_alpha_rows) * y_std + y_mean).detach()
    deep_test = (model(x_test, t_test, test_alpha_rows) * y_std + y_mean).detach()
    mlsl_test = torch.stack(
        [layer(u0, t, alpha) for u0, t, alpha in zip(test_u0_rows, test_t_rows, test_alpha_rows, strict=True)]
    ).detach()

    print("train_alpha_min:", train_alphas.min().item())
    print("train_alpha_max:", train_alphas.max().item())
    print("ood_alpha_min:", ood_alphas.min().item())
    print("ood_alpha_max:", ood_alphas.max().item())
    print("conditional_mlp_train_rel_error:", rel(deep_train, train_y))
    print("conditional_mlp_ood_alpha_rel_error:", rel(deep_test, test_y))
    print("mlsl_ood_alpha_rel_error:", rel(mlsl_test, test_y))


if __name__ == "__main__":
    main()

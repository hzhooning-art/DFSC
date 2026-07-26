"""Dataset-level long-time operator learning: MLSL primitive vs FNO baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import FNO1D, MittagLefflerSpectralLayer, dirichlet_laplacian_1d


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def make_random_initial_conditions(x: torch.Tensor, count: int, max_mode: int = 5) -> torch.Tensor:
    coeffs = torch.randn(count, max_mode) / torch.arange(1, max_mode + 1, dtype=x.dtype)
    fields = []
    for row in coeffs:
        u = torch.zeros_like(x)
        for k, c in enumerate(row, start=1):
            u = u + c * torch.sin(k * torch.pi * x)
        fields.append(u)
    fields = torch.stack(fields, dim=0)
    return fields / torch.linalg.norm(fields, dim=1, keepdim=True).clamp_min(1e-12)


def flatten_dataset(u0_batch: torch.Tensor, times: torch.Tensor, layer: MittagLefflerSpectralLayer, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u0_rows = []
    t_rows = []
    y_rows = []
    for u0 in u0_batch:
        y = layer(u0, times, alpha).detach()
        u0_rows.append(u0[None, :].expand(times.numel(), -1))
        t_rows.append(times)
        y_rows.append(y)
    return torch.cat(u0_rows, dim=0), torch.cat(t_rows, dim=0), torch.cat(y_rows, dim=0)


def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(41)

    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=64, num_modes=24)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    alpha_true = torch.tensor(1.42)
    train_times = torch.linspace(0.0, 0.04, 6)
    test_times = torch.linspace(0.06, 0.16, 8)

    train_u0 = make_random_initial_conditions(x, count=24)
    test_u0 = make_random_initial_conditions(x, count=8)
    train_u0_rows, train_t_rows, train_y = flatten_dataset(train_u0, train_times, layer, alpha_true)
    test_u0_rows, test_t_rows, test_y = flatten_dataset(test_u0, test_times, layer, alpha_true)

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

    fno = FNO1D(modes=12, width=32, layers=4).to(dtype=torch.float64)
    opt_fno = torch.optim.Adam(fno.parameters(), lr=2e-3)
    for _ in range(450):
        opt_fno.zero_grad()
        pred = fno(train_u0_rows, train_t_rows)
        loss = torch.mean((pred - train_y) ** 2)
        loss.backward()
        opt_fno.step()
    fno_train = fno(train_u0_rows, train_t_rows).detach()
    fno_test = fno(test_u0_rows, test_t_rows).detach()

    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est.item())
    print("mlsl_test_long_time_rel_error:", rel(mlsl_test, test_y))
    print("fno_train_rel_error:", rel(fno_train, train_y))
    print("fno_test_long_time_rel_error:", rel(fno_test, test_y))


if __name__ == "__main__":
    main()

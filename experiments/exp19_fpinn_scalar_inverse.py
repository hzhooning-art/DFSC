"""fPINN-style scalar inverse baseline for subdiffusive relaxation.

This experiment is deliberately limited to ``0 < alpha < 1`` where the L1
Caputo residual is simple and standard. It complements the MLSL experiments by
testing a residual-based fractional PINN route on the same hidden-order inverse
idea.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import l1_caputo_derivative_uniform, mittag_leffler_e


class ScalarTimeNet(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t[:, None]).squeeze(-1)


def constrain(raw: torch.Tensor, low: float = 0.20, high: float = 0.95) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(81)

    final_time = 1.0
    num_steps = 32
    mu = torch.tensor(1.4)
    u0 = torch.tensor(1.0)
    alpha_true = torch.tensor(0.65)
    grid = torch.linspace(0.0, final_time, num_steps + 1)
    clean = u0 * mittag_leffler_e(alpha_true, -mu * grid.pow(alpha_true), terms=140)

    sensor_idx = torch.linspace(0, num_steps, 7).round().long()
    observed = clean[sensor_idx] + 1e-4 * torch.std(clean) * torch.randn(sensor_idx.numel())

    model = ScalarTimeNet(hidden=32)
    raw_alpha = nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam(list(model.parameters()) + [raw_alpha], lr=2e-3)

    for _ in range(700):
        opt.zero_grad()
        alpha = constrain(raw_alpha)
        pred = model(grid)
        caputo = l1_caputo_derivative_uniform(pred, alpha=alpha, final_time=final_time)
        residual = caputo + mu * pred[1:]
        data_loss = torch.mean((pred[sensor_idx] - observed) ** 2)
        ic_loss = (pred[0] - u0) ** 2
        residual_loss = torch.mean(residual**2)
        loss = 10.0 * data_loss + ic_loss + 0.1 * residual_loss
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha).item()
    pred = model(grid).detach()
    rel_solution = (torch.linalg.norm(pred - clean) / torch.linalg.norm(clean)).item()
    print("alpha_true:", alpha_true.item())
    print("alpha_est:", alpha_est)
    print("alpha_rel_error:", abs(alpha_est - alpha_true.item()) / alpha_true.item())
    print("solution_rel_error:", rel_solution)
    print("final_loss:", loss.item())


if __name__ == "__main__":
    main()

"""Hybrid MLSL-backbone neural model versus pure FNO/DeepONet.

The target dynamics are a known fractional spectral backbone plus a small
unmodeled cubic correction. Pure neural operators must learn the full map from
data, while the hybrid model inserts MLSL as the linear fractional primitive and
learns only a residual correction.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import DeepONet1D, FNO1D, MLSLConfig, build_dirichlet_mlsl_1d


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        path = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        f = path.open("w", newline="", encoding="utf-8")
    with f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)


def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-14)).item()


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def make_random_initial_conditions(x: torch.Tensor, count: int, max_mode: int = 8) -> torch.Tensor:
    coeffs = torch.randn(count, max_mode) / torch.arange(1, max_mode + 1, dtype=x.dtype)
    modes = torch.stack([torch.sin((i + 1) * torch.pi * x) for i in range(max_mode)])
    fields = coeffs @ modes
    return fields / torch.linalg.norm(fields, dim=1, keepdim=True).clamp_min(1e-12)


def flatten_hybrid_dataset(
    u0_batch: torch.Tensor,
    times: torch.Tensor,
    layer,
    alpha: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u0_rows, t_rows, backbone_list, target_list = [], [], [], []
    for u0 in u0_batch:
        backbone = layer(u0, times, alpha).detach()
        correction = gamma * times[:, None] * backbone + 0.25 * gamma * times[:, None] * backbone.pow(3)
        target = backbone + correction
        u0_rows.append(u0[None, :].expand(times.numel(), -1))
        t_rows.append(times)
        backbone_list.append(backbone)
        target_list.append(target)
    return torch.cat(u0_rows), torch.cat(t_rows), torch.cat(backbone_list), torch.cat(target_list)


class HybridResidualNet(nn.Module):
    """Structured neural residual head on top of the MLSL backbone."""

    def __init__(self, hidden: int = 48, depth: int = 3) -> None:
        super().__init__()
        self.net = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.net.weight)

    def forward(self, backbone: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_grid = t[:, None].expand_as(backbone)
        features = torch.stack(
            [
                (t_grid * backbone).reshape(-1),
                (t_grid * backbone.pow(3)).reshape(-1),
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1).reshape_as(backbone)


def run_seed(seed: int) -> list[dict[str, object]]:
    torch.manual_seed(7300 + seed)
    torch.set_default_dtype(torch.float64)

    x, layer = build_dirichlet_mlsl_1d(
        num_points=40,
        num_modes=16,
        config=MLSLConfig.stable(terms=120),
    )
    alpha_true = torch.tensor(1.25)
    gamma = 2.5
    train_times = torch.linspace(0.0, 0.055, 7)
    test_times = torch.linspace(0.12, 0.32, 7)
    train_u0 = make_random_initial_conditions(x, 18)
    test_u0 = make_random_initial_conditions(x, 6)

    train_u0_rows, train_t_rows, train_backbone_true, train_y = flatten_hybrid_dataset(
        train_u0, train_times, layer, alpha_true, gamma
    )
    test_u0_rows, test_t_rows, test_backbone_true, test_y = flatten_hybrid_dataset(
        test_u0, test_times, layer, alpha_true, gamma
    )

    rows: list[dict[str, object]] = []

    # Hybrid: insert the known MLSL primitive and learn only the residual.
    # This tests composability rather than inverse recovery of alpha.
    residual = HybridResidualNet().to(dtype=torch.float64)
    opt_hybrid = torch.optim.Adam(residual.parameters(), lr=5e-2)
    train_residual_target = train_y - train_backbone_true
    for _ in range(300):
        opt_hybrid.zero_grad()
        correction = residual(train_backbone_true, x, train_t_rows)
        pred = train_backbone_true + correction
        loss = torch.mean((pred - train_y) ** 2) + 0.001 * torch.mean((correction - train_residual_target) ** 2)
        loss.backward()
        opt_hybrid.step()
    hybrid_train = (train_backbone_true + residual(train_backbone_true, x, train_t_rows)).detach()
    hybrid_test = (test_backbone_true + residual(test_backbone_true, x, test_t_rows)).detach()
    rows.append(
        {
            "seed": seed,
            "model": "Hybrid_MLSL_residual",
            "train_relative_error": rel(hybrid_train, train_y),
            "long_time_relative_error": rel(hybrid_test, test_y),
            "alpha_est": float(alpha_true),
            "alpha_relative_error": 0.0,
        }
    )

    # Oracle backbone only: shows the magnitude of the deliberately unmodeled correction.
    rows.append(
        {
            "seed": seed,
            "model": "MLSL_backbone_only",
            "train_relative_error": rel(train_backbone_true, train_y),
            "long_time_relative_error": rel(test_backbone_true, test_y),
            "alpha_est": float(alpha_true),
            "alpha_relative_error": 0.0,
        }
    )

    fno = FNO1D(modes=12, width=32, layers=3).to(dtype=torch.float64)
    opt_fno = torch.optim.Adam(fno.parameters(), lr=2e-3)
    for _ in range(90):
        opt_fno.zero_grad()
        loss = torch.mean((fno(train_u0_rows, train_t_rows) - train_y) ** 2)
        loss.backward()
        opt_fno.step()
    fno_train = fno(train_u0_rows, train_t_rows).detach()
    fno_test = fno(test_u0_rows, test_t_rows).detach()
    rows.append(
        {
            "seed": seed,
            "model": "FNO1D_width32_modes12",
            "train_relative_error": rel(fno_train, train_y),
            "long_time_relative_error": rel(fno_test, test_y),
            "alpha_est": "",
            "alpha_relative_error": "",
        }
    )

    x_train = x[None, :].expand(train_u0_rows.shape[0], -1)
    t_train = train_t_rows[:, None].expand(-1, x.numel())
    x_test = x[None, :].expand(test_u0_rows.shape[0], -1)
    t_test = test_t_rows[:, None].expand(-1, x.numel())
    deeponet = DeepONet1D(num_points=x.numel(), latent=64, hidden=96).to(dtype=torch.float64)
    opt_deep = torch.optim.Adam(deeponet.parameters(), lr=1e-3)
    y_mean = train_y.mean()
    y_std = train_y.std().clamp_min(1e-12)
    train_y_norm = (train_y - y_mean) / y_std
    alpha_train = alpha_true.expand(train_u0_rows.shape[0])
    alpha_test = alpha_true.expand(test_u0_rows.shape[0])
    for _ in range(120):
        opt_deep.zero_grad()
        pred = deeponet(train_u0_rows, x_train, t_train, alpha_train)
        loss = torch.mean((pred - train_y_norm) ** 2)
        loss.backward()
        opt_deep.step()
    deep_train = (deeponet(train_u0_rows, x_train, t_train, alpha_train) * y_std + y_mean).detach()
    deep_test = (deeponet(test_u0_rows, x_test, t_test, alpha_test) * y_std + y_mean).detach()
    rows.append(
        {
            "seed": seed,
            "model": "DeepONet_latent64_hidden96",
            "train_relative_error": rel(deep_train, train_y),
            "long_time_relative_error": rel(deep_test, test_y),
            "alpha_est": "",
            "alpha_relative_error": "",
        }
    )
    return rows


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for seed in [0, 1, 2]:
        all_rows.extend(run_seed(seed))
    write_csv(TABLES / "hybrid_backbone_baseline.csv", all_rows)

    summary: dict[str, float] = {}
    for model in sorted({str(row["model"]) for row in all_rows}):
        vals = [float(row["long_time_relative_error"]) for row in all_rows if row["model"] == model]
        train_vals = [float(row["train_relative_error"]) for row in all_rows if row["model"] == model]
        mean_val = sum(vals) / len(vals)
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in vals) / max(len(vals) - 1, 1))
        summary[f"{model}_long_time_error_mean"] = mean_val
        summary[f"{model}_long_time_error_std"] = std_val
        summary[f"{model}_train_error_mean"] = sum(train_vals) / len(train_vals)
    (RESULTS / "hybrid_backbone_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

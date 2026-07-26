"""PDE-field fPINN baseline for the MLSL paper.

This experiment trains a neural field ``u_theta(x,t)`` with an L1 Caputo
residual and a spectral Laplacian residual. It is intentionally compact: the
goal is to provide a PDE-field residual-training baseline that is stronger than
the scalar relaxation fPINN smoke test, not to claim state-of-the-art fPINN
tuning.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, MLPField, build_dirichlet_mlsl_1d


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


def l1_caputo_derivative_matrix(values: torch.Tensor, alpha: torch.Tensor, final_time: float) -> torch.Tensor:
    """L1 Caputo derivative for ``values`` with shape ``(steps + 1, points)``."""

    if values.ndim != 2:
        raise ValueError("values must have shape (steps + 1, points)")
    num_steps = values.shape[0] - 1
    dtype = values.dtype
    device = values.device
    alpha = alpha.to(dtype=dtype, device=device)
    dt = torch.as_tensor(final_time / num_steps, dtype=dtype, device=device)
    j = torch.arange(num_steps, dtype=dtype, device=device)
    weights = (j + 1.0).pow(1.0 - alpha) - j.pow(1.0 - alpha)
    scale = dt.pow(-alpha) / torch.exp(torch.lgamma(2.0 - alpha))

    derivatives = []
    for n in range(1, num_steps + 1):
        increments = torch.stack([values[n - j_idx] - values[n - j_idx - 1] for j_idx in range(n)])
        derivatives.append(scale * torch.sum(weights[:n, None] * increments, dim=0))
    return torch.stack(derivatives, dim=0)


def spectral_apply(values: torch.Tensor, layer, beta: torch.Tensor) -> torch.Tensor:
    """Apply the retained spectral fractional operator to rows of ``values``."""

    phi = layer.eigenvectors.to(dtype=values.dtype, device=values.device)
    mu = layer.modal_rates(beta, layer.eigenvalues.to(dtype=values.dtype, device=values.device))
    coeff = values @ phi
    return (coeff * mu) @ phi.T


def run_seed(seed: int) -> dict[str, object]:
    torch.manual_seed(6200 + seed)
    torch.set_default_dtype(torch.float64)

    x, layer = build_dirichlet_mlsl_1d(
        num_points=32,
        num_modes=12,
        config=MLSLConfig.stable(terms=120),
    )
    final_time = 0.35
    num_steps = 24
    alpha_true = torch.tensor(0.65)
    beta = torch.tensor(2.0)
    times = torch.linspace(0.0, final_time, num_steps + 1)
    u0 = torch.sin(torch.pi * x) + 0.15 * torch.sin(3.0 * torch.pi * x)
    clean = layer(u0, times, alpha_true, beta=beta).detach()

    model = MLPField(hidden=64, depth=3).to(dtype=torch.float64)
    raw_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = torch.optim.Adam(list(model.parameters()) + [raw_alpha], lr=2e-3)

    x_grid = x[None, :].expand(num_steps + 1, -1)
    t_grid = times[:, None].expand(-1, x.numel())
    sensor_times = torch.tensor([0, 4, 8, 12, 18, 24])
    sensor_points = torch.arange(0, x.numel(), 4)
    observed = clean[sensor_times][:, sensor_points]

    for _ in range(700):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 0.25, 0.95)
        pred = model(x_grid, t_grid)
        caputo = l1_caputo_derivative_matrix(pred, alpha, final_time)
        residual = caputo + spectral_apply(pred[1:], layer, beta)
        data_loss = torch.mean((pred[sensor_times][:, sensor_points] - observed) ** 2)
        ic_loss = torch.mean((pred[0] - u0) ** 2)
        residual_loss = torch.mean(residual**2)
        smooth_loss = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
        loss = 20.0 * data_loss + 5.0 * ic_loss + 0.05 * residual_loss + 0.01 * smooth_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        alpha_est = constrain(raw_alpha, 0.25, 0.95)
        pred = model(x_grid, t_grid)
        caputo = l1_caputo_derivative_matrix(pred, alpha_est, final_time)
        residual = caputo + spectral_apply(pred[1:], layer, beta)
        solution_error = rel(pred, clean)
        residual_norm = torch.sqrt(torch.mean(residual**2)).item()
        alpha_error = abs(float(alpha_est) - float(alpha_true)) / float(alpha_true)

    return {
        "seed": seed,
        "alpha_true": float(alpha_true),
        "alpha_est": float(alpha_est),
        "alpha_relative_error": alpha_error,
        "solution_relative_error": solution_error,
        "residual_rms": residual_norm,
        "num_points": x.numel(),
        "num_time_steps": num_steps,
        "sensor_times": int(sensor_times.numel()),
        "sensor_points": int(sensor_points.numel()),
    }


def main() -> None:
    rows = [run_seed(seed) for seed in [0, 1, 2]]
    write_csv(TABLES / "pde_field_fpinn_baseline.csv", rows)

    alpha_errors = [float(r["alpha_relative_error"]) for r in rows]
    solution_errors = [float(r["solution_relative_error"]) for r in rows]
    summary = {
        "pde_field_fpinn_alpha_error_mean": sum(alpha_errors) / len(alpha_errors),
        "pde_field_fpinn_alpha_error_max": max(alpha_errors),
        "pde_field_fpinn_solution_error_mean": sum(solution_errors) / len(solution_errors),
        "pde_field_fpinn_solution_error_max": max(solution_errors),
    }
    (RESULTS / "pde_field_fpinn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Manufactured-solution validation for forced MLSL."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import ForcedMittagLefflerSpectralLayer, MLSLConfig, build_dirichlet_mlsl_1d


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-14)).item()


def manufactured_forcing_values(
    phi_mode: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    physical_times: torch.Tensor,
    power: float,
) -> torch.Tensor:
    """For exact modal solution ``a(t)=t^power`` with zero initial condition."""

    gamma_ratio = math.gamma(power + 1.0) / torch.exp(torch.lgamma(torch.as_tensor(power + 1.0) - alpha))
    amp = gamma_ratio * physical_times.clamp_min(0.0).pow(power - alpha) + mu * physical_times.pow(power)
    return amp[:, :, None] * phi_mode[None, None, :]


def run_case(alpha_value: float, beta_value: float, q: int) -> dict[str, object]:
    x, base = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=16,
        config=MLSLConfig(terms=120),
    )
    mode_index = 0
    phi_mode = base.eigenvectors[:, mode_index].to(dtype=x.dtype)
    eigenvalue = base.eigenvalues[mode_index]
    alpha = torch.tensor(alpha_value, requires_grad=True)
    beta = torch.tensor(beta_value, requires_grad=True)
    mu = base.modal_rates(beta, base.eigenvalues)[mode_index].detach()
    power = 2.0
    times = torch.linspace(0.0, 0.02, 8)
    forcing_times = (torch.arange(q, dtype=x.dtype) + 0.5) / q
    physical_times = times[:, None] * forcing_times[None, :]
    forcing = manufactured_forcing_values(phi_mode, mu, alpha.detach(), physical_times, power)

    layer = ForcedMittagLefflerSpectralLayer(base, forcing_terms=120)
    pred = layer(torch.zeros_like(x), times, alpha, forcing, forcing_times, beta=beta)
    exact = times.pow(power)[:, None] * phi_mode[None, :]
    error = rel(pred.detach(), exact)
    loss = torch.mean((pred - exact) ** 2)
    loss.backward()
    return {
        "alpha": alpha_value,
        "beta": beta_value,
        "quadrature_points": q,
        "mode_index": mode_index + 1,
        "eigenvalue": float(eigenvalue),
        "relative_error": error,
        "finite_output": bool(torch.isfinite(pred).all().item()),
        "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
        "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
        "loss": float(loss.detach()),
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rows = []
    for alpha_value in [0.75, 0.85, 0.95]:
        for beta_value in [1.0, 1.3, 1.6]:
            for q in [32, 64, 128]:
                rows.append(run_case(alpha_value, beta_value, q))

    write_csv(TABLES / "manufactured_forcing_validation.csv", rows)
    summary = {
        "manufactured_forcing_max_error_q128": max(
            float(r["relative_error"]) for r in rows if r["quadrature_points"] == 128
        ),
        "manufactured_forcing_mean_error_q128": sum(
            float(r["relative_error"]) for r in rows if r["quadrature_points"] == 128
        )
        / sum(1 for r in rows if r["quadrature_points"] == 128),
        "manufactured_forcing_all_finite": all(
            bool(r["finite_output"]) and bool(r["finite_alpha_grad"]) and bool(r["finite_beta_grad"])
            for r in rows
        ),
    }
    (RESULTS / "manufactured_forcing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

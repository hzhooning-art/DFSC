"""Supplemental reliability experiments for MLSL paper review concerns.

This experiment targets two reviewer-facing questions:

1. Is the branch-aware Mittag-Leffler evaluator reliable in the spectral region
   used by the layer?
2. Is alpha/beta inverse recovery robust under noise and sparse observations?
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import MittagLefflerSpectralLayer, dirichlet_laplacian_1d, mittag_leffler_e

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def mp_mittag_leffler(alpha: float, z: float, *, dps: int = 100) -> float:
    mp.mp.dps = dps
    a = mp.mpf(alpha)
    zz = mp.mpf(z)
    total = mp.mpf("0")
    for k in range(20000):
        term = zz**k / mp.gamma(a * k + 1)
        total += term
        if abs(term) < mp.mpf("1e-80"):
            break
    return float(total)


def rel_err(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-14)


def evaluator_reference_and_finiteness() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alphas = [0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65]
    z_values = [-0.0, -1.0, -2.0, -4.0, -6.0, -8.0, -10.0, -20.0, -40.0, -80.0]
    reference_z_limit = 8.0

    for alpha_value in alphas:
        alpha = torch.tensor(alpha_value)
        for z_value in z_values:
            z = torch.tensor(z_value)
            series = mittag_leffler_e(alpha, z, terms=160, method="series").item()
            hybrid = mittag_leffler_e(alpha, z, terms=160, method="hybrid").item()
            has_reference = abs(z_value) <= reference_z_limit
            reference = mp_mittag_leffler(alpha_value, z_value) if has_reference else float("nan")
            rows.append(
                {
                    "alpha": alpha_value,
                    "z": z_value,
                    "has_reference": has_reference,
                    "reference": reference,
                    "series_value": series,
                    "hybrid_value": hybrid,
                    "series_is_finite": bool(torch.isfinite(torch.tensor(series)).item()),
                    "hybrid_is_finite": bool(torch.isfinite(torch.tensor(hybrid)).item()),
                    "series_relative_error": rel_err(series, reference) if has_reference else float("nan"),
                    "hybrid_relative_error": rel_err(hybrid, reference) if has_reference else float("nan"),
                }
            )

    referenced = [r for r in rows if r["has_reference"]]
    tail = [r for r in rows if not r["has_reference"]]
    summary = {
        "evaluator_reference_cases": len(referenced),
        "evaluator_reference_max_hybrid_relative_error": max(r["hybrid_relative_error"] for r in referenced),
        "evaluator_reference_max_series_relative_error": max(r["series_relative_error"] for r in referenced),
        "evaluator_tail_cases": len(tail),
        "evaluator_tail_hybrid_finite_rate": sum(1 for r in tail if r["hybrid_is_finite"]) / len(tail),
        "evaluator_tail_series_finite_rate": sum(1 for r in tail if r["series_is_finite"]) / len(tail),
        "evaluator_series_large_error_cases": sum(
            1 for r in referenced if r["series_relative_error"] > 1e-3
        ),
    }
    return rows, summary


def evaluator_gradient_check() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        (0.65, -2.0),
        (0.65, -6.0),
        (0.65, -20.0),
        (1.25, -2.0),
        (1.25, -6.0),
        (1.25, -20.0),
        (1.65, -2.0),
        (1.65, -6.0),
        (1.65, -20.0),
    ]
    eps = 1e-4
    for alpha_value, z_value in cases:
        alpha = torch.tensor(alpha_value, requires_grad=True)
        z = torch.tensor(z_value)
        value = mittag_leffler_e(alpha, z, terms=160, method="hybrid")
        value.backward()
        grad_auto = alpha.grad.detach().item()
        plus = mittag_leffler_e(torch.tensor(alpha_value + eps), z, terms=160, method="hybrid").item()
        minus = mittag_leffler_e(torch.tensor(alpha_value - eps), z, terms=160, method="hybrid").item()
        grad_fd = (plus - minus) / (2.0 * eps)
        rows.append(
            {
                "alpha": alpha_value,
                "z": z_value,
                "value": value.detach().item(),
                "grad_auto": grad_auto,
                "grad_fd": grad_fd,
                "relative_error": rel_err(grad_auto, grad_fd),
                "finite_grad": bool(torch.isfinite(torch.tensor(grad_auto)).item()),
            }
        )

    summary = {
        "evaluator_gradient_cases": len(rows),
        "evaluator_gradient_finite_rate": sum(1 for r in rows if r["finite_grad"]) / len(rows),
        "evaluator_gradient_max_relative_error": max(r["relative_error"] for r in rows),
        "evaluator_gradient_median_relative_error": sorted(r["relative_error"] for r in rows)[len(rows) // 2],
    }
    return rows, summary


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def make_inverse_problem() -> tuple[torch.Tensor, MittagLefflerSpectralLayer, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=18)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=120)
    u0 = (
        torch.sin(torch.pi * x)
        + 0.25 * torch.sin(2.0 * torch.pi * x)
        + 0.12 * torch.sin(5.0 * torch.pi * x)
    )
    alpha_true = torch.tensor(1.38)
    beta_true = torch.tensor(1.35)
    return x, layer, u0, alpha_true, beta_true


def run_inverse_case(
    *,
    seed: int,
    noise_level: float,
    num_sensors: int,
    num_times: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    x, layer, u0, alpha_true, beta_true = make_inverse_problem()
    full_times = torch.linspace(0.0, 0.03, 10)
    time_idx = torch.linspace(0, full_times.numel() - 1, num_times).round().long()
    sensor_idx = torch.linspace(4, x.numel() - 5, num_sensors).round().long()
    times = full_times[time_idx]
    clean = layer(u0, times, alpha_true, beta=beta_true).detach()
    scale = torch.std(clean).clamp_min(1e-12)
    observed = clean[:, sensor_idx] + noise_level * scale * torch.randn(num_times, num_sensors)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
    raw_beta = torch.nn.Parameter(torch.tensor(0.5))
    opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
    for _ in range(500):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        pred = layer(u0, times, alpha, beta=beta)[:, sensor_idx]
        loss = torch.mean((pred - observed) ** 2)
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    return {
        "seed": seed,
        "noise_level": noise_level,
        "num_sensors": num_sensors,
        "num_times": num_times,
        "alpha_est": alpha_est,
        "beta_est": beta_est,
        "alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
        "beta_relative_error": abs(beta_est - beta_true.item()) / beta_true.item(),
        "final_loss": loss.item(),
    }


def inverse_robustness_multiseed() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    noise_levels = [0.0, 1e-3, 1e-2, 5e-2]
    observation_configs = [(128, 8), (16, 6), (8, 4), (4, 4)]
    seeds = [0, 1, 2, 3, 4]

    for noise_level in noise_levels:
        for num_sensors, num_times in observation_configs:
            for seed in seeds:
                rows.append(
                    run_inverse_case(
                        seed=5000 + seed,
                        noise_level=noise_level,
                        num_sensors=num_sensors,
                        num_times=num_times,
                    )
                )

    aggregate_rows: list[dict[str, Any]] = []
    for noise_level in noise_levels:
        for num_sensors, num_times in observation_configs:
            group = [
                r
                for r in rows
                if r["noise_level"] == noise_level
                and r["num_sensors"] == num_sensors
                and r["num_times"] == num_times
            ]
            alpha_errors = torch.tensor([r["alpha_relative_error"] for r in group])
            beta_errors = torch.tensor([r["beta_relative_error"] for r in group])
            aggregate_rows.append(
                {
                    "noise_level": noise_level,
                    "num_sensors": num_sensors,
                    "num_times": num_times,
                    "seeds": len(group),
                    "alpha_error_mean": alpha_errors.mean().item(),
                    "alpha_error_std": alpha_errors.std(unbiased=True).item(),
                    "beta_error_mean": beta_errors.mean().item(),
                    "beta_error_std": beta_errors.std(unbiased=True).item(),
                    "joint_error_mean": torch.maximum(alpha_errors, beta_errors).mean().item(),
                    "joint_error_max": torch.maximum(alpha_errors, beta_errors).max().item(),
                }
            )

    dense_low_noise = next(
        r for r in aggregate_rows if r["noise_level"] == 1e-3 and r["num_sensors"] == 128 and r["num_times"] == 8
    )
    sparse_high_noise = next(
        r for r in aggregate_rows if r["noise_level"] == 5e-2 and r["num_sensors"] == 4 and r["num_times"] == 4
    )
    summary = {
        "inverse_robustness_cases": len(rows),
        "inverse_robustness_aggregate_cases": len(aggregate_rows),
        "inverse_dense_noise_1e_minus_3_joint_error_mean": dense_low_noise["joint_error_mean"],
        "inverse_sparse_high_noise_joint_error_mean": sparse_high_noise["joint_error_mean"],
        "inverse_sparse_high_noise_joint_error_max": sparse_high_noise["joint_error_max"],
    }
    return rows, aggregate_rows, summary


def main() -> None:
    torch.set_default_dtype(torch.float64)
    ensure_dirs()

    evaluator_rows, evaluator_summary = evaluator_reference_and_finiteness()
    evaluator_grad_rows, evaluator_grad_summary = evaluator_gradient_check()
    inverse_rows, inverse_aggregate_rows, inverse_summary = inverse_robustness_multiseed()

    write_csv(TABLES / "evaluator_reliability_grid.csv", evaluator_rows)
    write_csv(TABLES / "evaluator_gradient_reliability.csv", evaluator_grad_rows)
    write_csv(TABLES / "inverse_robustness_multiseed_raw.csv", inverse_rows)
    write_csv(TABLES / "inverse_robustness_multiseed_summary.csv", inverse_aggregate_rows)

    summary = {}
    summary.update(evaluator_summary)
    summary.update(evaluator_grad_summary)
    summary.update(inverse_summary)
    write_json(RESULTS / "reliability_and_inverse_robustness_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

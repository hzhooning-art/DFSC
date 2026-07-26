"""Reviewer-facing transition and inverse-baseline diagnostics.

This experiment adds two compact checks that support the MLSL primitive claim:

1. The hybrid Mittag-Leffler evaluator is probed near the branch transition.
2. The trainable-order inverse task is compared with the PDE-field fPINN
   baseline under the same synthetic fractional diffusion setting.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, build_dirichlet_mlsl_1d, mittag_leffler_e

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-14)).item()


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def transition_smoothness_probe() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha_value, center in [(0.55, 4.0), (0.65, 8.0), (1.25, 8.0)]:
        for z_abs in torch.linspace(center * 0.75, center * 1.25, 21):
            alpha = torch.tensor(alpha_value, requires_grad=True)
            z = -z_abs.clone().detach()
            value = mittag_leffler_e(alpha, z, terms=160, method="hybrid")
            value.backward()
            rows.append(
                {
                    "alpha": alpha_value,
                    "z": float(z),
                    "value": float(value.detach()),
                    "grad_alpha": float(alpha.grad.detach()),
                    "finite_value": bool(torch.isfinite(value.detach()).item()),
                    "finite_grad": bool(torch.isfinite(alpha.grad.detach()).item()),
                }
            )

    jumps = []
    grad_jumps = []
    for alpha_value in sorted({float(row["alpha"]) for row in rows}):
        group = [row for row in rows if float(row["alpha"]) == alpha_value]
        group.sort(key=lambda row: float(row["z"]))
        for left, right in zip(group, group[1:]):
            jumps.append(abs(float(right["value"]) - float(left["value"])))
            grad_jumps.append(abs(float(right["grad_alpha"]) - float(left["grad_alpha"])))

    summary = {
        "transition_probe_cases": len(rows),
        "transition_finite_value_rate": sum(1 for row in rows if row["finite_value"]) / len(rows),
        "transition_finite_grad_rate": sum(1 for row in rows if row["finite_grad"]) / len(rows),
        "transition_max_adjacent_value_jump": max(jumps),
        "transition_max_adjacent_grad_jump": max(grad_jumps),
    }
    return rows, summary


def mlsl_inverse_same_setting(seed: int) -> dict[str, Any]:
    torch.manual_seed(7300 + seed)
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

    sensor_times = torch.tensor([0, 4, 8, 12, 18, 24])
    sensor_points = torch.arange(0, x.numel(), 4)
    observed = clean[sensor_times][:, sensor_points]

    raw_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = torch.optim.Adam([raw_alpha], lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 0.25, 0.95)
        pred = layer(u0, times, alpha, beta=beta)
        loss = torch.mean((pred[sensor_times][:, sensor_points] - observed) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        alpha_est = constrain(raw_alpha, 0.25, 0.95)
        pred = layer(u0, times, alpha_est, beta=beta)
        solution_error = rel(pred, clean)
        alpha_error = abs(float(alpha_est) - float(alpha_true)) / float(alpha_true)

    return {
        "model": "MLSL direct inverse",
        "seed": seed,
        "alpha_true": float(alpha_true),
        "alpha_est": float(alpha_est),
        "alpha_relative_error": alpha_error,
        "solution_relative_error": solution_error,
        "num_points": int(x.numel()),
        "num_time_steps": num_steps,
        "sensor_times": int(sensor_times.numel()),
        "sensor_points": int(sensor_points.numel()),
    }


def same_setting_inverse_baseline() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mlsl_rows = [mlsl_inverse_same_setting(seed) for seed in [0, 1, 2]]
    fpinn_path = TABLES / "pde_field_fpinn_baseline.csv"
    fpinn_rows: list[dict[str, Any]] = []
    if fpinn_path.exists():
        with fpinn_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["model"] = "PDE-field fPINN"
                fpinn_rows.append(row)

    rows = mlsl_rows + fpinn_rows
    aggregate_rows: list[dict[str, Any]] = []
    for model in ["MLSL direct inverse", "PDE-field fPINN"]:
        group = [row for row in rows if row["model"] == model]
        alpha_errors = [float(row["alpha_relative_error"]) for row in group]
        solution_errors = [float(row["solution_relative_error"]) for row in group]
        aggregate_rows.append(
            {
                "model": model,
                "seeds": len(group),
                "alpha_error_mean": mean(alpha_errors),
                "alpha_error_max": max(alpha_errors),
                "solution_error_mean": mean(solution_errors),
                "solution_error_max": max(solution_errors),
            }
        )

    mlsl_alpha = next(row for row in aggregate_rows if row["model"] == "MLSL direct inverse")
    fpinn_alpha = next(row for row in aggregate_rows if row["model"] == "PDE-field fPINN")
    summary = {
        "same_setting_inverse_rows": len(rows),
        "mlsl_inverse_alpha_error_mean": mlsl_alpha["alpha_error_mean"],
        "fpinn_inverse_alpha_error_mean": fpinn_alpha["alpha_error_mean"],
        "mlsl_vs_fpinn_alpha_error_ratio": fpinn_alpha["alpha_error_mean"] / max(mlsl_alpha["alpha_error_mean"], 1e-14),
        "mlsl_inverse_solution_error_mean": mlsl_alpha["solution_error_mean"],
        "fpinn_inverse_solution_error_mean": fpinn_alpha["solution_error_mean"],
    }
    return rows, aggregate_rows, summary


def main() -> None:
    torch.set_default_dtype(torch.float64)
    TABLES.mkdir(parents=True, exist_ok=True)

    transition_rows, transition_summary = transition_smoothness_probe()
    inverse_rows, inverse_aggregate_rows, inverse_summary = same_setting_inverse_baseline()

    write_csv(TABLES / "transition_smoothness_probe.csv", transition_rows)
    write_csv(TABLES / "same_setting_inverse_baseline_raw.csv", inverse_rows)
    write_csv(TABLES / "same_setting_inverse_baseline_summary.csv", inverse_aggregate_rows)

    summary = {}
    summary.update(transition_summary)
    summary.update(inverse_summary)
    write_json(RESULTS / "transition_and_inverse_baseline_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

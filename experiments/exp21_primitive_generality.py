"""Generality matrix for the MLSL primitive.

This experiment checks whether the current component behaves like a reusable
SciML primitive across dimensions, fractional orders, initial conditions,
batching, differentiability, and inverse recovery.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import MLSLConfig, build_dirichlet_mlsl_1d, build_dirichlet_mlsl_2d


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        f = fallback.open("w", newline="", encoding="utf-8")
        path = fallback
    with f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-14)).item()


def make_1d_fields(x: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    return [
        ("low_mode", torch.sin(torch.pi * x)),
        ("multi_mode", torch.sin(torch.pi * x) + 0.25 * torch.sin(3.0 * torch.pi * x)),
        (
            "localized",
            torch.exp(-120.0 * (x - 0.35).square()) - 0.35 * torch.exp(-80.0 * (x - 0.75).square()),
        ),
    ]


def forward_gradient_matrix() -> list[dict[str, object]]:
    x, layer_1d = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=20,
        config=MLSLConfig.stable(terms=120),
    )
    times = torch.linspace(0.0, 0.035, 7)
    rows: list[dict[str, object]] = []
    for field_name, u0 in make_1d_fields(x):
        for alpha_value in [0.65, 1.20, 1.75]:
            for beta_value in [1.00, 1.50, 2.00]:
                alpha = torch.tensor(alpha_value, requires_grad=True)
                beta = torch.tensor(beta_value, requires_grad=True)
                out = layer_1d(u0, times, alpha, beta=beta)
                loss = out[-1].square().mean() + 0.1 * out.mean().square()
                loss.backward()
                rows.append(
                    {
                        "check": "1d_forward_gradient",
                        "dimension": "1d",
                        "field": field_name,
                        "alpha": alpha_value,
                        "beta": beta_value,
                        "output_shape": str(tuple(out.shape)),
                        "finite_output": bool(torch.isfinite(out).all().item()),
                        "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                        "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                        "metric": float(loss.detach()),
                        "passed": bool(
                            torch.isfinite(out).all().item()
                            and torch.isfinite(alpha.grad).item()
                            and torch.isfinite(beta.grad).item()
                        ),
                    }
                )
    return rows


def linearity_and_batch_checks() -> list[dict[str, object]]:
    x, layer = build_dirichlet_mlsl_1d(
        num_points=80,
        num_modes=16,
        config=MLSLConfig.stable(terms=120),
    )
    times = torch.linspace(0.0, 0.03, 5)
    u = torch.sin(torch.pi * x) + 0.1 * torch.sin(4.0 * torch.pi * x)
    v = torch.sin(2.0 * torch.pi * x) - 0.2 * torch.sin(5.0 * torch.pi * x)
    a = torch.tensor(1.7)
    b = torch.tensor(-0.4)
    alpha = torch.tensor(1.35)
    beta = torch.tensor(1.45)
    lhs = layer(a * u + b * v, times, alpha, beta=beta)
    rhs = a * layer(u, times, alpha, beta=beta) + b * layer(v, times, alpha, beta=beta)
    linearity_error = rel_error(lhs, rhs)

    batch = torch.stack([u, v, a * u + b * v])
    batch_out = layer(batch, times, alpha, beta=beta)
    batch_error = rel_error(batch_out[2], lhs)
    return [
        {
            "check": "linearity",
            "dimension": "1d",
            "field": "combined",
            "alpha": float(alpha),
            "beta": float(beta),
            "output_shape": str(tuple(lhs.shape)),
            "finite_output": bool(torch.isfinite(lhs).all().item()),
            "finite_alpha_grad": True,
            "finite_beta_grad": True,
            "metric": linearity_error,
            "passed": linearity_error < 1e-12,
        },
        {
            "check": "batch_consistency",
            "dimension": "1d",
            "field": "three_fields",
            "alpha": float(alpha),
            "beta": float(beta),
            "output_shape": str(tuple(batch_out.shape)),
            "finite_output": bool(torch.isfinite(batch_out).all().item()),
            "finite_alpha_grad": True,
            "finite_beta_grad": True,
            "metric": batch_error,
            "passed": batch_error < 1e-12,
        },
    ]


def two_dimensional_check() -> list[dict[str, object]]:
    coords, layer = build_dirichlet_mlsl_2d(
        num_points_1d=14,
        num_modes_1d=5,
        config=MLSLConfig.stable(terms=120),
    )
    x = coords[:, 0]
    y = coords[:, 1]
    u0 = (
        torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        + 0.15 * torch.sin(2.0 * torch.pi * x) * torch.sin(3.0 * torch.pi * y)
    )
    rows: list[dict[str, object]] = []
    for alpha_value, beta_value in [(0.75, 1.2), (1.35, 1.5), (1.75, 1.9)]:
        alpha = torch.tensor(alpha_value, requires_grad=True)
        beta = torch.tensor(beta_value, requires_grad=True)
        out = layer(u0, torch.linspace(0.0, 0.018, 5), alpha, beta=beta)
        loss = out.square().mean()
        loss.backward()
        rows.append(
            {
                "check": "2d_forward_gradient",
                "dimension": "2d",
                "field": "tensor_product_modes",
                "alpha": alpha_value,
                "beta": beta_value,
                "output_shape": str(tuple(out.shape)),
                "finite_output": bool(torch.isfinite(out).all().item()),
                "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                "metric": float(loss.detach()),
                "passed": bool(
                    torch.isfinite(out).all().item()
                    and torch.isfinite(alpha.grad).item()
                    and torch.isfinite(beta.grad).item()
                ),
            }
        )
    return rows


def inverse_recovery_checks() -> list[dict[str, object]]:
    torch.manual_seed(23)
    x, layer = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=18,
        config=MLSLConfig.stable(terms=120),
    )
    times = torch.linspace(0.0, 0.025, 7)
    rows: list[dict[str, object]] = []
    cases = [
        ("low_mode_alpha_only", torch.sin(torch.pi * x), 1.25, 1.20, "alpha_only"),
        (
            "multi_mode_joint",
            torch.sin(torch.pi * x) + 0.2 * torch.sin(4.0 * torch.pi * x),
            1.45,
            1.55,
            "joint_alpha_beta",
        ),
    ]
    for field_name, u0, alpha_true_value, beta_true_value, task in cases:
        alpha_true = torch.tensor(alpha_true_value)
        beta_true = torch.tensor(beta_true_value)
        observed = layer(u0, times, alpha_true, beta=beta_true).detach()
        observed = observed + 1e-5 * torch.std(observed) * torch.randn_like(observed)

        raw_alpha = torch.nn.Parameter(torch.tensor(0.3))
        raw_beta = torch.nn.Parameter(torch.tensor(0.2))
        params = [raw_alpha] if task == "alpha_only" else [raw_alpha, raw_beta]
        opt = torch.optim.Adam(params, lr=0.04)
        for _ in range(420):
            opt.zero_grad()
            alpha = constrain(raw_alpha, 0.45, 1.95)
            beta = beta_true if task == "alpha_only" else constrain(raw_beta, 0.70, 2.00)
            pred = layer(u0, times, alpha, beta=beta)
            loss = torch.mean((pred - observed) ** 2)
            loss.backward()
            opt.step()

        alpha_est = constrain(raw_alpha, 0.45, 1.95).detach()
        if task == "alpha_only":
            beta_est = beta_true
            err = abs(float(alpha_est) - alpha_true_value) / alpha_true_value
        else:
            beta_est = constrain(raw_beta, 0.70, 2.00).detach()
            err = max(
                abs(float(alpha_est) - alpha_true_value) / alpha_true_value,
                abs(float(beta_est) - beta_true_value) / beta_true_value,
            )
        rows.append(
            {
                "check": task,
                "dimension": "1d",
                "field": field_name,
                "alpha": alpha_true_value,
                "beta": beta_true_value,
                "output_shape": str(tuple(observed.shape)),
                "finite_output": True,
                "finite_alpha_grad": True,
                "finite_beta_grad": True,
                "metric": err,
                "passed": err < 0.05,
            }
        )
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(forward_gradient_matrix())
    rows.extend(linearity_and_batch_checks())
    rows.extend(two_dimensional_check())
    rows.extend(inverse_recovery_checks())

    write_csv(TABLES / "primitive_generality_matrix.csv", rows)
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    finite_metrics = [
        float(row["metric"])
        for row in rows
        if math.isfinite(float(row["metric"]))
    ]
    failed = [row for row in rows if not row["passed"]]
    summary = {
        "primitive_generality_total_checks": total,
        "primitive_generality_passed_checks": passed,
        "primitive_generality_failed_checks": len(failed),
        "primitive_generality_pass_rate": passed / total,
        "primitive_generality_max_finite_metric": max(finite_metrics),
        "primitive_generality_nonfinite_metric_count": total - len(finite_metrics),
        "primitive_generality_failed_cases": [
            {
                "check": row["check"],
                "dimension": row["dimension"],
                "field": row["field"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "metric": row["metric"],
            }
            for row in failed
        ],
    }
    (RESULTS / "primitive_generality_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

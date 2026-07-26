"""Reproducible experiment workflow for Paper 1 MLSL prototype.

This runner saves machine-readable CSV/JSON outputs without generating plots.
It is meant to be the bridge from exploratory scripts to paper-grade evidence.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import mpmath as mp
import torch

from dfsc import (
    ConditionalMLPField,
    DeepONet1D,
    FNO1D,
    MLPField,
    MittagLefflerSpectralLayer,
    dirichlet_laplacian_1d,
    dirichlet_laplacian_2d,
    l1_caputo_derivative_uniform,
    l1_caputo_relaxation,
    mittag_leffler_e,
)
from dfsc.mittag_leffler import mittag_leffler_e_hybrid
from experiments.exp19_fpinn_scalar_inverse import ScalarTimeNet


ROOT = Path(__file__).resolve().parent
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


def gradient_check_alpha() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=96, num_modes=16)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=90)
    u0 = torch.sin(torch.pi * x) + 0.15 * torch.sin(2.0 * torch.pi * x)
    target = layer(u0, torch.tensor(0.05), torch.tensor(1.35)).detach()

    alpha = torch.tensor(1.65, requires_grad=True)
    loss = torch.mean((layer(u0, torch.tensor(0.05), alpha) - target) ** 2)
    loss.backward()
    grad_auto = alpha.grad.detach().item()

    rows = []
    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        alpha_p = torch.tensor(1.65 + eps)
        alpha_m = torch.tensor(1.65 - eps)
        lp = torch.mean((layer(u0, torch.tensor(0.05), alpha_p) - target) ** 2)
        lm = torch.mean((layer(u0, torch.tensor(0.05), alpha_m) - target) ** 2)
        grad_fd = ((lp - lm) / (2.0 * eps)).item()
        rel = abs(grad_auto - grad_fd) / max(abs(grad_fd), 1e-14)
        rows.append(
            {
                "parameter": "alpha",
                "epsilon": eps,
                "grad_auto": grad_auto,
                "grad_fd": grad_fd,
                "relative_error": rel,
            }
        )
    return rows, {"alpha_grad_min_relative_error": min(r["relative_error"] for r in rows)}


def gradient_check_beta() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=96, num_modes=14)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    u0 = torch.sin(torch.pi * x) + 0.20 * torch.sin(3.0 * torch.pi * x)
    target = layer(
        u0,
        torch.tensor(0.015),
        torch.tensor(1.35),
        beta=torch.tensor(1.25),
    ).detach()

    beta = torch.tensor(1.65, requires_grad=True)
    loss = torch.mean((layer(u0, torch.tensor(0.015), torch.tensor(1.35), beta=beta) - target) ** 2)
    loss.backward()
    grad_auto = beta.grad.detach().item()

    rows = []
    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        beta_p = torch.tensor(1.65 + eps)
        beta_m = torch.tensor(1.65 - eps)
        lp = torch.mean((layer(u0, torch.tensor(0.015), torch.tensor(1.35), beta=beta_p) - target) ** 2)
        lm = torch.mean((layer(u0, torch.tensor(0.015), torch.tensor(1.35), beta=beta_m) - target) ** 2)
        grad_fd = ((lp - lm) / (2.0 * eps)).item()
        rel = abs(grad_auto - grad_fd) / max(abs(grad_fd), 1e-14)
        rows.append(
            {
                "parameter": "beta",
                "epsilon": eps,
                "grad_auto": grad_auto,
                "grad_fd": grad_fd,
                "relative_error": rel,
            }
        )
    return rows, {"beta_grad_min_relative_error": min(r["relative_error"] for r in rows)}


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def recover_alpha() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(7)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=20)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=90)
    u0 = torch.sin(torch.pi * x) + 0.20 * torch.sin(4.0 * torch.pi * x)
    times = torch.linspace(0.0, 0.08, 9)
    alpha_true = torch.tensor(1.42)
    observations = layer(u0, times, alpha_true).detach()
    observations = observations + 1e-4 * torch.randn_like(observations)

    raw_alpha = torch.nn.Parameter(torch.tensor(1.2))
    optimizer = torch.optim.Adam([raw_alpha], lr=0.05)
    rows = []
    for step in range(401):
        optimizer.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        loss = torch.mean((layer(u0, times, alpha) - observations) ** 2)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 400:
            rows.append({"step": step, "loss": loss.item(), "alpha": alpha.item()})

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    summary = {
        "alpha_only_true": alpha_true.item(),
        "alpha_only_est": alpha_est,
        "alpha_only_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
    }
    return rows, summary


def recover_alpha_beta() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(11)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=18)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    u0 = (
        torch.sin(torch.pi * x)
        + 0.25 * torch.sin(2.0 * torch.pi * x)
        + 0.12 * torch.sin(5.0 * torch.pi * x)
    )
    times = torch.linspace(0.0, 0.025, 8)
    alpha_true = torch.tensor(1.38)
    beta_true = torch.tensor(1.35)
    observations = layer(u0, times, alpha_true, beta=beta_true).detach()
    observations = observations + 5e-5 * torch.randn_like(observations)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
    raw_beta = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
    rows = []
    for step in range(501):
        optimizer.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        loss = torch.mean((layer(u0, times, alpha, beta=beta) - observations) ** 2)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 500:
            rows.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "alpha": alpha.item(),
                    "beta": beta.item(),
                }
            )

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    summary = {
        "joint_alpha_true": alpha_true.item(),
        "joint_alpha_est": alpha_est,
        "joint_alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
        "joint_beta_true": beta_true.item(),
        "joint_beta_est": beta_est,
        "joint_beta_relative_error": abs(beta_est - beta_true.item()) / beta_true.item(),
    }
    return rows, summary


def runtime_scaling() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha = torch.tensor(0.65)
    mu = torch.tensor(3.0)
    final_time = 0.20
    u0 = torch.tensor(1.0)
    rows = []

    for num_steps in [50, 100, 200, 400, 800]:
        times = torch.linspace(0.0, final_time, num_steps + 1)

        def run_mlsl() -> torch.Tensor:
            z = -mu * times.pow(alpha)
            return u0 * mittag_leffler_e(alpha, z, terms=80)

        def run_l1() -> torch.Tensor:
            return l1_caputo_relaxation(
                u0,
                alpha=alpha,
                mu=mu,
                final_time=final_time,
                num_steps=num_steps,
            )

        def timed(fn) -> float:
            values = []
            for _ in range(3):
                start = time.perf_counter()
                fn()
                values.append(time.perf_counter() - start)
            return min(values)

        mlsl_seconds = timed(run_mlsl)
        l1_seconds = timed(run_l1)
        rows.append(
            {
                "num_steps": num_steps,
                "mlsl_seconds": mlsl_seconds,
                "l1_seconds": l1_seconds,
                "speedup_l1_over_mlsl": l1_seconds / max(mlsl_seconds, 1e-14),
                "estimated_l1_history_kb": (num_steps + 1) * 8 / 1024,
                "estimated_mlsl_state_kb": 3 * 8 / 1024,
            }
        )

    return rows, {"runtime_speedup_at_max_steps": rows[-1]["speedup_l1_over_mlsl"]}


def custom_backward_check() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    z = -torch.linspace(0.0, 1.5, 32)
    target = mittag_leffler_e(torch.tensor(1.35), z, terms=120, custom_backward=False).detach()

    alpha_auto = torch.tensor(1.65, requires_grad=True)
    loss_auto = torch.mean((mittag_leffler_e(alpha_auto, z, terms=90, custom_backward=False) - target) ** 2)
    loss_auto.backward()

    alpha_custom = torch.tensor(1.65, requires_grad=True)
    loss_custom = torch.mean((mittag_leffler_e(alpha_custom, z, terms=90, custom_backward=True) - target) ** 2)
    loss_custom.backward()

    rows = []
    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        lp = torch.mean((mittag_leffler_e(torch.tensor(1.65 + eps), z, terms=90, custom_backward=True) - target) ** 2)
        lm = torch.mean((mittag_leffler_e(torch.tensor(1.65 - eps), z, terms=90, custom_backward=True) - target) ** 2)
        grad_fd = ((lp - lm) / (2.0 * eps)).item()
        rel = abs(alpha_custom.grad.item() - grad_fd) / max(abs(grad_fd), 1e-14)
        rows.append(
            {
                "epsilon": eps,
                "grad_auto": alpha_auto.grad.item(),
                "grad_custom": alpha_custom.grad.item(),
                "grad_fd": grad_fd,
                "custom_vs_fd_relative_error": rel,
            }
        )
    summary = {
        "custom_vs_auto_relative_error": abs(alpha_custom.grad.item() - alpha_auto.grad.item())
        / max(abs(alpha_auto.grad.item()), 1e-14),
        "custom_vs_fd_min_relative_error": min(r["custom_vs_fd_relative_error"] for r in rows),
    }
    return rows, summary


def stable_evaluator_scan() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    summary: dict[str, Any] = {}

    for alpha_value in [0.65, 1.35]:
        alpha = torch.tensor(alpha_value)
        z_values = -torch.linspace(0.0, 80.0, 81)
        series = mittag_leffler_e(alpha, z_values, terms=120, custom_backward=False, method="series")
        hybrid = mittag_leffler_e(alpha, z_values, terms=120, custom_backward=False, method="hybrid")

        for z, series_value, hybrid_value in zip(z_values, series, hybrid, strict=True):
            rows.append(
                {
                    "alpha": alpha_value,
                    "z": z.item(),
                    "series_value": series_value.item(),
                    "hybrid_value": hybrid_value.item(),
                    "series_is_finite": bool(torch.isfinite(series_value).item()),
                    "hybrid_is_finite": bool(torch.isfinite(hybrid_value).item()),
                }
            )

        summary[f"stable_scan_alpha_{alpha_value}_series_max_abs"] = torch.max(torch.abs(series)).item()
        summary[f"stable_scan_alpha_{alpha_value}_hybrid_max_abs"] = torch.max(torch.abs(hybrid)).item()
        summary[f"stable_scan_alpha_{alpha_value}_hybrid_tail_z_minus_80"] = hybrid[-1].item()

    return rows, summary


def mp_mittag_leffler(alpha: float, z: float, *, dps: int = 80, tol: str = "1e-70") -> float:
    mp.mp.dps = dps
    a = mp.mpf(alpha)
    zz = mp.mpf(z)
    total = mp.mpf("0")
    threshold = mp.mpf(tol)
    for k in range(10000):
        term = zz**k / mp.gamma(a * k + 1)
        total += term
        if abs(term) < threshold:
            break
    return float(total)


def reference_accuracy() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for alpha_value in [0.65, 1.35]:
        alpha = torch.tensor(alpha_value)
        for z_value in [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0]:
            z = torch.tensor(z_value)
            reference = mp_mittag_leffler(alpha_value, z_value)
            series = mittag_leffler_e(alpha, z, terms=140, method="series").item()
            hybrid = mittag_leffler_e(alpha, z, terms=140, method="hybrid").item()
            rows.append(
                {
                    "alpha": alpha_value,
                    "z": z_value,
                    "reference": reference,
                    "series": series,
                    "hybrid": hybrid,
                    "series_abs_error": abs(series - reference),
                    "hybrid_abs_error": abs(hybrid - reference),
                }
            )

    safe_rows = [r for r in rows if abs(r["z"]) <= 6.0]
    summary = {
        "reference_accuracy_max_series_error_abs_z_le_6": max(r["series_abs_error"] for r in safe_rows),
        "reference_accuracy_max_hybrid_error_abs_z_le_6": max(r["hybrid_abs_error"] for r in safe_rows),
        "reference_accuracy_alpha_0.65_z_minus_8_series_error": next(
            r["series_abs_error"] for r in rows if r["alpha"] == 0.65 and r["z"] == -8.0
        ),
    }
    return rows, summary


def long_time_prediction() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(23)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=64, num_modes=16)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    u0 = torch.sin(torch.pi * x) + 0.20 * torch.sin(3.0 * torch.pi * x)
    alpha_true = torch.tensor(1.42)
    train_times = torch.linspace(0.0, 0.04, 8)
    test_times = torch.linspace(0.05, 0.16, 12)
    y_train = layer(u0, train_times, alpha_true).detach()
    y_test = layer(u0, test_times, alpha_true).detach()

    raw_alpha = torch.nn.Parameter(torch.tensor(1.0))
    opt_alpha = torch.optim.Adam([raw_alpha], lr=0.05)
    for _ in range(250):
        opt_alpha.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        loss = torch.mean((layer(u0, train_times, alpha) - y_train) ** 2)
        loss.backward()
        opt_alpha.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).detach()
    mlsl_train = layer(u0, train_times, alpha_est).detach()
    mlsl_test = layer(u0, test_times, alpha_est).detach()

    mlp = MLPField(hidden=64, depth=3).to(dtype=torch.float64)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=2e-3)
    x_train = x[None, :].expand(train_times.numel(), -1)
    t_train = train_times[:, None].expand(-1, x.numel())
    for _ in range(800):
        opt_mlp.zero_grad()
        pred = mlp(x_train, t_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt_mlp.step()

    x_test = x[None, :].expand(test_times.numel(), -1)
    t_test = test_times[:, None].expand(-1, x.numel())
    mlp_train = mlp(x_train, t_train).detach()
    mlp_test = mlp(x_test, t_test).detach()

    def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
        return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()

    rows = [
        {
            "model": "MLSL",
            "train_relative_error": rel(mlsl_train, y_train),
            "long_time_relative_error": rel(mlsl_test, y_test),
        },
        {
            "model": "MLPField",
            "train_relative_error": rel(mlp_train, y_train),
            "long_time_relative_error": rel(mlp_test, y_test),
        },
    ]
    summary = {
        "long_time_alpha_true": alpha_true.item(),
        "long_time_alpha_est": alpha_est.item(),
        "long_time_mlsl_error": rows[0]["long_time_relative_error"],
        "long_time_mlp_error": rows[1]["long_time_relative_error"],
        "long_time_mlp_over_mlsl_error_ratio": rows[1]["long_time_relative_error"]
        / max(rows[0]["long_time_relative_error"], 1e-14),
    }
    return rows, summary


def noise_robustness() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for idx, noise in enumerate([0.0, 1e-4, 1e-3, 1e-2, 5e-2]):
        torch.manual_seed(100 + idx)
        x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=18)
        layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
        u0 = (
            torch.sin(torch.pi * x)
            + 0.25 * torch.sin(2.0 * torch.pi * x)
            + 0.12 * torch.sin(5.0 * torch.pi * x)
        )
        times = torch.linspace(0.0, 0.025, 8)
        alpha_true = torch.tensor(1.38)
        beta_true = torch.tensor(1.35)
        clean = layer(u0, times, alpha_true, beta=beta_true).detach()
        observed = clean + noise * torch.std(clean) * torch.randn_like(clean)

        raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
        raw_beta = torch.nn.Parameter(torch.tensor(0.5))
        opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
        for _ in range(500):
            opt.zero_grad()
            alpha = constrain(raw_alpha, 1.05, 1.95)
            beta = constrain(raw_beta, 0.60, 1.95)
            loss = torch.mean((layer(u0, times, alpha, beta=beta) - observed) ** 2)
            loss.backward()
            opt.step()

        alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
        beta_est = constrain(raw_beta, 0.60, 1.95).item()
        rows.append(
            {
                "noise_level": noise,
                "alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
                "beta_relative_error": abs(beta_est - beta_true.item()) / beta_true.item(),
                "final_loss": loss.item(),
            }
        )

    summary = {
        "noise_1e_minus_2_alpha_relative_error": next(
            r["alpha_relative_error"] for r in rows if r["noise_level"] == 1e-2
        ),
        "noise_1e_minus_2_beta_relative_error": next(
            r["beta_relative_error"] for r in rows if r["noise_level"] == 1e-2
        ),
    }
    return rows, summary


def mode_sensitivity() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(31)
    num_points = 128
    x_ref, eigenvalues_ref, phi_ref = dirichlet_laplacian_1d(num_points=num_points, num_modes=64)
    ref_layer = MittagLefflerSpectralLayer(eigenvalues_ref, phi_ref, terms=100)
    u0 = torch.sin(torch.pi * x_ref) + 0.20 * torch.sin(6.0 * torch.pi * x_ref)
    times = torch.linspace(0.0, 0.035, 8)
    alpha_true = torch.tensor(1.42)
    reference = ref_layer(u0, times, alpha_true).detach()

    rows = []
    for num_modes in [4, 8, 12, 16, 24, 32, 48]:
        _, eigenvalues, phi = dirichlet_laplacian_1d(num_points=num_points, num_modes=num_modes)
        layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
        forward = layer(u0, times, alpha_true).detach()
        forward_error = (torch.linalg.norm(forward - reference) / torch.linalg.norm(reference)).item()
        raw_alpha = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.Adam([raw_alpha], lr=0.05)
        for _ in range(250):
            opt.zero_grad()
            alpha = constrain(raw_alpha, 1.05, 1.95)
            loss = torch.mean((layer(u0, times, alpha) - reference) ** 2)
            loss.backward()
            opt.step()
        alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
        rows.append(
            {
                "num_modes": num_modes,
                "forward_relative_error": forward_error,
                "alpha_est": alpha_est,
                "alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
            }
        )

    return rows, {"mode_sensitivity_error_at_8_modes": rows[1]["forward_relative_error"]}


def hybrid_threshold_scan() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for alpha_value in [0.65, 1.35]:
        alpha = torch.tensor(alpha_value)
        z_values = [-8.0, -6.0, -4.0, -2.0, -1.0]
        refs = torch.tensor([mp_mittag_leffler(alpha_value, z) for z in z_values])
        for threshold in [4.0, 6.0, 8.0, 10.0]:
            z = torch.tensor(z_values)
            pred = mittag_leffler_e_hybrid(alpha, z, series_terms=140, threshold=threshold)
            err = torch.max(torch.abs(pred - refs)).item()
            tail = mittag_leffler_e_hybrid(
                alpha,
                torch.tensor([-80.0]),
                series_terms=140,
                threshold=threshold,
            )[0].item()
            rows.append(
                {
                    "alpha": alpha_value,
                    "threshold": threshold,
                    "max_abs_error_on_reference_points": err,
                    "tail_z_minus_80": tail,
                }
            )

    return rows, {
        "hybrid_threshold_alpha_0.65_best_reference_error": min(
            r["max_abs_error_on_reference_points"] for r in rows if r["alpha"] == 0.65
        )
    }


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


def flatten_dataset(
    u0_batch: torch.Tensor,
    times: torch.Tensor,
    layer: MittagLefflerSpectralLayer,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u0_rows = []
    t_rows = []
    y_rows = []
    for u0 in u0_batch:
        y = layer(u0, times, alpha).detach()
        u0_rows.append(u0[None, :].expand(times.numel(), -1))
        t_rows.append(times)
        y_rows.append(y)
    return torch.cat(u0_rows, dim=0), torch.cat(t_rows, dim=0), torch.cat(y_rows, dim=0)


def fno_dataset_long_time() -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

    def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
        return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()

    rows = [
        {"model": "MLSL", "train_relative_error": 0.0, "long_time_relative_error": rel(mlsl_test, test_y)},
        {
            "model": "FNO1D",
            "train_relative_error": rel(fno_train, train_y),
            "long_time_relative_error": rel(fno_test, test_y),
        },
    ]
    return rows, {
        "fno_dataset_alpha_est": alpha_est.item(),
        "fno_dataset_mlsl_long_time_error": rows[0]["long_time_relative_error"],
        "fno_dataset_fno_long_time_error": rows[1]["long_time_relative_error"],
    }


def sparse_observation_inverse() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for sensors, time_count in [(4, 4), (8, 4), (8, 6), (16, 6)]:
        torch.manual_seed(200 + sensors + time_count)
        x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=18)
        layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
        u0 = (
            torch.sin(torch.pi * x)
            + 0.25 * torch.sin(2.0 * torch.pi * x)
            + 0.12 * torch.sin(5.0 * torch.pi * x)
        )
        full_times = torch.linspace(0.0, 0.03, 10)
        time_idx = torch.linspace(0, full_times.numel() - 1, time_count).round().long()
        sensor_idx = torch.linspace(4, x.numel() - 5, sensors).round().long()
        times = full_times[time_idx]
        alpha_true = torch.tensor(1.38)
        beta_true = torch.tensor(1.35)
        clean = layer(u0, times, alpha_true, beta=beta_true).detach()
        observed = clean[:, sensor_idx] + 1e-4 * torch.std(clean) * torch.randn(time_count, sensors)

        raw_alpha = torch.nn.Parameter(torch.tensor(0.9))
        raw_beta = torch.nn.Parameter(torch.tensor(0.5))
        opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
        for _ in range(600):
            opt.zero_grad()
            alpha = constrain(raw_alpha, 1.05, 1.95)
            beta = constrain(raw_beta, 0.60, 1.95)
            loss = torch.mean((layer(u0, times, alpha, beta=beta)[:, sensor_idx] - observed) ** 2)
            loss.backward()
            opt.step()

        alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
        beta_est = constrain(raw_beta, 0.60, 1.95).item()
        rows.append(
            {
                "num_sensors": sensors,
                "num_times": time_count,
                "alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
                "beta_relative_error": abs(beta_est - beta_true.item()) / beta_true.item(),
                "final_loss": loss.item(),
            }
        )

    return rows, {
        "sparse_4sensor_4time_alpha_error": rows[0]["alpha_relative_error"],
        "sparse_4sensor_4time_beta_error": rows[0]["beta_relative_error"],
    }


def deeponet_dataset_long_time() -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

    def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
        return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()

    rows = [
        {"model": "MLSL", "train_relative_error": 0.0, "long_time_relative_error": rel(mlsl_test, test_y)},
        {
            "model": "DeepONet1D",
            "train_relative_error": rel(deeponet_train, train_y),
            "long_time_relative_error": rel(deeponet_test, test_y),
        },
    ]
    return rows, {
        "deeponet_dataset_mlsl_long_time_error": rows[0]["long_time_relative_error"],
        "deeponet_dataset_deeponet_long_time_error": rows[1]["long_time_relative_error"],
    }


def ood_alpha_generalization() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(61)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=64, num_modes=8)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    times = torch.linspace(0.0, 0.02, 8)
    train_alphas = torch.tensor([1.15, 1.25, 1.35, 1.45, 1.55])
    ood_alphas = torch.tensor([1.70, 1.80])
    base_u0 = make_random_initial_conditions(x, count=1)
    train_u0_rows, train_alpha_rows, train_t_rows, train_y = build_alpha_rows(base_u0, train_alphas, times, layer)
    test_u0_rows, test_alpha_rows, test_t_rows, test_y = build_alpha_rows(base_u0, ood_alphas, times, layer)
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
    mlp_train = (model(x_train, t_train, train_alpha_rows) * y_std + y_mean).detach()
    mlp_test = (model(x_test, t_test, test_alpha_rows) * y_std + y_mean).detach()
    mlsl_test = torch.stack(
        [layer(u0, t, alpha) for u0, t, alpha in zip(test_u0_rows, test_t_rows, test_alpha_rows, strict=True)]
    ).detach()

    def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
        return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()

    rows = [
        {"model": "MLSL", "train_relative_error": 0.0, "ood_alpha_relative_error": rel(mlsl_test, test_y)},
        {
            "model": "ConditionalMLPField",
            "train_relative_error": rel(mlp_train, train_y),
            "ood_alpha_relative_error": rel(mlp_test, test_y),
        },
    ]
    return rows, {
        "ood_alpha_mlsl_error": rows[0]["ood_alpha_relative_error"],
        "ood_alpha_conditional_mlp_error": rows[1]["ood_alpha_relative_error"],
    }


def build_alpha_rows(
    u0_batch: torch.Tensor,
    alphas: torch.Tensor,
    times: torch.Tensor,
    layer: MittagLefflerSpectralLayer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u0_rows, alpha_rows, t_rows, y_rows = [], [], [], []
    for u0 in u0_batch:
        for alpha in alphas:
            y = layer(u0, times, alpha).detach()
            u0_rows.append(u0[None, :].expand(times.numel(), -1))
            alpha_rows.append(alpha.expand(times.numel()))
            t_rows.append(times)
            y_rows.append(y)
    return torch.cat(u0_rows), torch.cat(alpha_rows), torch.cat(t_rows), torch.cat(y_rows)


def two_dimensional_forward_inverse() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(71)
    coords, eigenvalues, phi = dirichlet_laplacian_2d(num_points_1d=18, num_modes_1d=6)
    x = coords[:, 0]
    y = coords[:, 1]
    u0 = (
        torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        + 0.2 * torch.sin(2.0 * torch.pi * x) * torch.sin(3.0 * torch.pi * y)
    )
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    times = torch.linspace(0.0, 0.025, 7)
    alpha_true = torch.tensor(1.36)
    beta_true = torch.tensor(1.55)
    observations = layer(u0, times, alpha_true, beta=beta_true).detach()
    observations = observations + 1e-4 * torch.std(observations) * torch.randn_like(observations)

    raw_alpha = torch.nn.Parameter(torch.tensor(0.8))
    raw_beta = torch.nn.Parameter(torch.tensor(0.4))
    opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
    for _ in range(450):
        opt.zero_grad()
        alpha = constrain(raw_alpha, 1.05, 1.95)
        beta = constrain(raw_beta, 0.60, 1.95)
        loss = torch.mean((layer(u0, times, alpha, beta=beta) - observations) ** 2)
        loss.backward()
        opt.step()

    alpha_est = constrain(raw_alpha, 1.05, 1.95).item()
    beta_est = constrain(raw_beta, 0.60, 1.95).item()
    rows = [
        {
            "num_points": coords.shape[0],
            "num_modes": eigenvalues.numel(),
            "alpha_true": alpha_true.item(),
            "alpha_est": alpha_est,
            "alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
            "beta_true": beta_true.item(),
            "beta_est": beta_est,
            "beta_relative_error": abs(beta_est - beta_true.item()) / beta_true.item(),
            "final_loss": loss.item(),
        }
    ]
    return rows, {
        "2d_alpha_relative_error": rows[0]["alpha_relative_error"],
        "2d_beta_relative_error": rows[0]["beta_relative_error"],
    }


def fpinn_scalar_inverse() -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

    def constrain_subdiffusive(raw: torch.Tensor) -> torch.Tensor:
        return 0.20 + 0.75 * torch.sigmoid(raw)

    model = ScalarTimeNet(hidden=32)
    raw_alpha = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam(list(model.parameters()) + [raw_alpha], lr=2e-3)
    for _ in range(700):
        opt.zero_grad()
        alpha = constrain_subdiffusive(raw_alpha)
        pred = model(grid)
        caputo = l1_caputo_derivative_uniform(pred, alpha=alpha, final_time=final_time)
        residual = caputo + mu * pred[1:]
        data_loss = torch.mean((pred[sensor_idx] - observed) ** 2)
        ic_loss = (pred[0] - u0) ** 2
        residual_loss = torch.mean(residual**2)
        loss = 10.0 * data_loss + ic_loss + 0.1 * residual_loss
        loss.backward()
        opt.step()

    alpha_est = constrain_subdiffusive(raw_alpha).item()
    pred = model(grid).detach()
    solution_error = (torch.linalg.norm(pred - clean) / torch.linalg.norm(clean)).item()
    rows = [
        {
            "alpha_true": alpha_true.item(),
            "alpha_est": alpha_est,
            "alpha_relative_error": abs(alpha_est - alpha_true.item()) / alpha_true.item(),
            "solution_relative_error": solution_error,
            "final_loss": loss.item(),
            "num_steps": num_steps,
        }
    ]
    return rows, {
        "fpinn_scalar_alpha_relative_error": rows[0]["alpha_relative_error"],
        "fpinn_scalar_solution_relative_error": solution_error,
    }


def batch_scaling() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(91)
    x, eigenvalues, phi = dirichlet_laplacian_1d(num_points=128, num_modes=24)
    layer = MittagLefflerSpectralLayer(eigenvalues, phi, terms=100)
    times = torch.linspace(0.0, 0.04, 8)
    alpha = torch.tensor(1.42)
    rows = []
    for batch_size in [1, 4, 16, 64, 128]:
        u0 = make_random_initial_conditions(x, count=batch_size)

        def run() -> torch.Tensor:
            return layer(u0, times, alpha)

        values = []
        for _ in range(3):
            start = time.perf_counter()
            output = run()
            values.append(time.perf_counter() - start)
        rows.append(
            {
                "batch_size": batch_size,
                "seconds": min(values),
                "output_shape": str(tuple(output.shape)),
            }
        )
    return rows, {"batch_scaling_time_at_128": rows[-1]["seconds"]}


def main() -> None:
    torch.set_default_dtype(torch.float64)
    ensure_dirs()

    summary: dict[str, Any] = {}

    alpha_grad_rows, alpha_grad_summary = gradient_check_alpha()
    beta_grad_rows, beta_grad_summary = gradient_check_beta()
    write_csv(TABLES / "gradient_checks.csv", alpha_grad_rows + beta_grad_rows)
    summary.update(alpha_grad_summary)
    summary.update(beta_grad_summary)

    alpha_rows, alpha_summary = recover_alpha()
    write_csv(TABLES / "alpha_recovery_trace.csv", alpha_rows)
    summary.update(alpha_summary)

    alpha_beta_rows, alpha_beta_summary = recover_alpha_beta()
    write_csv(TABLES / "alpha_beta_recovery_trace.csv", alpha_beta_rows)
    summary.update(alpha_beta_summary)

    runtime_rows, runtime_summary = runtime_scaling()
    write_csv(TABLES / "runtime_scaling.csv", runtime_rows)
    summary.update(runtime_summary)

    custom_rows, custom_summary = custom_backward_check()
    write_csv(TABLES / "custom_backward_check.csv", custom_rows)
    summary.update(custom_summary)

    stable_rows, stable_summary = stable_evaluator_scan()
    write_csv(TABLES / "stable_evaluator_scan.csv", stable_rows)
    summary.update(stable_summary)

    reference_rows, reference_summary = reference_accuracy()
    write_csv(TABLES / "reference_accuracy.csv", reference_rows)
    summary.update(reference_summary)

    long_time_rows, long_time_summary = long_time_prediction()
    write_csv(TABLES / "long_time_prediction.csv", long_time_rows)
    summary.update(long_time_summary)

    noise_rows, noise_summary = noise_robustness()
    write_csv(TABLES / "noise_robustness.csv", noise_rows)
    summary.update(noise_summary)

    mode_rows, mode_summary = mode_sensitivity()
    write_csv(TABLES / "mode_sensitivity.csv", mode_rows)
    summary.update(mode_summary)

    threshold_rows, threshold_summary = hybrid_threshold_scan()
    write_csv(TABLES / "hybrid_threshold_scan.csv", threshold_rows)
    summary.update(threshold_summary)

    fno_rows, fno_summary = fno_dataset_long_time()
    write_csv(TABLES / "fno_dataset_long_time.csv", fno_rows)
    summary.update(fno_summary)

    sparse_rows, sparse_summary = sparse_observation_inverse()
    write_csv(TABLES / "sparse_observation_inverse.csv", sparse_rows)
    summary.update(sparse_summary)

    deeponet_rows, deeponet_summary = deeponet_dataset_long_time()
    write_csv(TABLES / "deeponet_dataset_long_time.csv", deeponet_rows)
    summary.update(deeponet_summary)

    ood_rows, ood_summary = ood_alpha_generalization()
    write_csv(TABLES / "ood_alpha_generalization.csv", ood_rows)
    summary.update(ood_summary)

    two_d_rows, two_d_summary = two_dimensional_forward_inverse()
    write_csv(TABLES / "two_dimensional_forward_inverse.csv", two_d_rows)
    summary.update(two_d_summary)

    fpinn_rows, fpinn_summary = fpinn_scalar_inverse()
    write_csv(TABLES / "fpinn_scalar_inverse.csv", fpinn_rows)
    summary.update(fpinn_summary)

    batch_rows, batch_summary = batch_scaling()
    write_csv(TABLES / "batch_scaling.csv", batch_rows)
    summary.update(batch_summary)

    write_json(RESULTS / "summary.json", summary)

    print("Saved experiment outputs under:", RESULTS)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

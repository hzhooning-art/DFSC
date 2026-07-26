"""Paper-grade extension experiments for the MLSL primitive."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import (
    DeepONet1D,
    FNO1D,
    ForcedMittagLefflerSpectralLayer,
    MLSLConfig,
    build_dirichlet_mlsl_1d,
    l1_caputo_derivative_uniform,
    l1_caputo_relaxation,
    mittag_leffler_e,
)
from experiments.exp19_fpinn_scalar_inverse import ScalarTimeNet


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
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


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def constrain(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-14)).item()


def make_random_initial_conditions(x: torch.Tensor, count: int, max_mode: int = 6) -> torch.Tensor:
    coeffs = torch.randn(count, max_mode) / torch.arange(1, max_mode + 1, dtype=x.dtype)
    modes = torch.stack([torch.sin((i + 1) * torch.pi * x) for i in range(max_mode)])
    fields = coeffs @ modes
    return fields / torch.linalg.norm(fields, dim=1, keepdim=True).clamp_min(1e-12)


def flatten_dataset(
    u0_batch: torch.Tensor,
    times: torch.Tensor,
    layer,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u0_rows, t_rows, y_rows = [], [], []
    for u0 in u0_batch:
        y = layer(u0, times, alpha).detach()
        u0_rows.append(u0[None, :].expand(times.numel(), -1))
        t_rows.append(times)
        y_rows.append(y)
    return torch.cat(u0_rows), torch.cat(t_rows), torch.cat(y_rows)


def long_horizon_stability_accuracy() -> tuple[list[dict[str, object]], dict[str, object]]:
    x, stable = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=20,
        config=MLSLConfig.stable(terms=120),
    )
    _, reference = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=20,
        config=MLSLConfig.stable(terms=180),
    )
    u0 = torch.sin(torch.pi * x) + 0.2 * torch.sin(3.0 * torch.pi * x)
    times = torch.linspace(0.0, 1.0, 21)
    rows = []
    for alpha_value in [0.65, 1.20, 1.75]:
        for beta_value in [1.0, 1.5, 2.0]:
            alpha = torch.tensor(alpha_value, requires_grad=True)
            beta = torch.tensor(beta_value, requires_grad=True)
            pred = stable(u0, times, alpha, beta=beta)
            ref = reference(u0, times, alpha.detach(), beta=beta.detach()).detach()
            loss = pred.square().mean()
            loss.backward()
            rows.append(
                {
                    "alpha": alpha_value,
                    "beta": beta_value,
                    "horizon": float(times[-1]),
                    "relative_difference_terms_120_vs_180": rel(pred.detach(), ref),
                    "finite_output": bool(torch.isfinite(pred).all().item()),
                    "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                    "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                    "passed": bool(
                        torch.isfinite(pred).all().item()
                        and torch.isfinite(alpha.grad).item()
                        and torch.isfinite(beta.grad).item()
                    ),
                }
            )
    return rows, {
        "long_horizon_max_term_difference": max(
            float(r["relative_difference_terms_120_vs_180"]) for r in rows
        ),
        "long_horizon_pass_rate": sum(1 for r in rows if r["passed"]) / len(rows),
    }


def multi_seed_inverse() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        x, layer = build_dirichlet_mlsl_1d(
            num_points=96,
            num_modes=18,
            config=MLSLConfig.stable(terms=120),
        )
        u0 = torch.sin(torch.pi * x) + 0.2 * torch.sin(4.0 * torch.pi * x)
        times = torch.linspace(0.0, 0.035, 9)
        alpha_true = torch.tensor(1.42)
        beta_true = torch.tensor(1.45)
        observed = layer(u0, times, alpha_true, beta=beta_true).detach()
        observed = observed + 1e-4 * torch.std(observed) * torch.randn_like(observed)
        raw_alpha = torch.nn.Parameter(torch.tensor(0.4))
        raw_beta = torch.nn.Parameter(torch.tensor(0.1))
        opt = torch.optim.Adam([raw_alpha, raw_beta], lr=0.04)
        for _ in range(420):
            opt.zero_grad()
            alpha = constrain(raw_alpha, 0.6, 1.95)
            beta = constrain(raw_beta, 0.7, 2.0)
            loss = torch.mean((layer(u0, times, alpha, beta=beta) - observed) ** 2)
            loss.backward()
            opt.step()
        alpha_est = float(constrain(raw_alpha, 0.6, 1.95).detach())
        beta_est = float(constrain(raw_beta, 0.7, 2.0).detach())
        rows.append(
            {
                "seed": seed,
                "alpha_true": float(alpha_true),
                "alpha_est": alpha_est,
                "alpha_relative_error": abs(alpha_est - float(alpha_true)) / float(alpha_true),
                "beta_true": float(beta_true),
                "beta_est": beta_est,
                "beta_relative_error": abs(beta_est - float(beta_true)) / float(beta_true),
                "final_loss": float(loss.detach()),
            }
        )
    alpha_errors = [float(r["alpha_relative_error"]) for r in rows]
    beta_errors = [float(r["beta_relative_error"]) for r in rows]
    alpha_mean, alpha_std = mean_std(alpha_errors)
    beta_mean, beta_std = mean_std(beta_errors)
    return rows, {
        "multi_seed_alpha_error_mean": alpha_mean,
        "multi_seed_alpha_error_std": alpha_std,
        "multi_seed_beta_error_mean": beta_mean,
        "multi_seed_beta_error_std": beta_std,
    }


def batch_and_device_profile() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        x, layer = build_dirichlet_mlsl_1d(
            num_points=128,
            num_modes=24,
            config=MLSLConfig.stable(terms=120, device=device),
        )
        x = x.to(device)
        layer = layer.to(device)
        times = torch.linspace(0.0, 0.06, 10, device=device)
        alpha = torch.tensor(1.35, device=device)
        beta = torch.tensor(1.45, device=device)
        for batch_size in [1, 16, 64, 256, 512]:
            torch.manual_seed(2000 + batch_size)
            u0 = make_random_initial_conditions(x.cpu(), batch_size).to(device)
            for _ in range(2):
                layer(u0, times, alpha, beta=beta)
            if device == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                out = layer(u0, times, alpha, beta=beta)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            rows.append(
                {
                    "device": device,
                    "batch_size": batch_size,
                    "seconds_min": min(timings),
                    "seconds_per_sample": min(timings) / batch_size,
                    "output_shape": str(tuple(out.shape)),
                    "finite_output": bool(torch.isfinite(out).all().item()),
                }
            )
    return rows, {
        "gpu_available": torch.cuda.is_available(),
        "batch_profile_cpu_seconds_at_512": [
            r["seconds_min"] for r in rows if r["device"] == "cpu" and r["batch_size"] == 512
        ][0],
    }


def stronger_operator_baselines() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for seed in [0, 1]:
        torch.manual_seed(3000 + seed)
        x, layer = build_dirichlet_mlsl_1d(
            num_points=64,
            num_modes=24,
            config=MLSLConfig.stable(terms=120),
        )
        alpha_true = torch.tensor(1.42)
        train_times = torch.linspace(0.0, 0.05, 7)
        test_times = torch.linspace(0.12, 0.30, 7)
        train_u0 = make_random_initial_conditions(x, 32)
        test_u0 = make_random_initial_conditions(x, 10)
        train_u0_rows, train_t_rows, train_y = flatten_dataset(train_u0, train_times, layer, alpha_true)
        test_u0_rows, test_t_rows, test_y = flatten_dataset(test_u0, test_times, layer, alpha_true)

        fno = FNO1D(modes=16, width=48, layers=4).to(dtype=torch.float64)
        opt_fno = torch.optim.Adam(fno.parameters(), lr=2e-3)
        for _ in range(350):
            opt_fno.zero_grad()
            loss = torch.mean((fno(train_u0_rows, train_t_rows) - train_y) ** 2)
            loss.backward()
            opt_fno.step()
        fno_train = fno(train_u0_rows, train_t_rows).detach()
        fno_test = fno(test_u0_rows, test_t_rows).detach()

        x_train = x[None, :].expand(train_u0_rows.shape[0], -1)
        x_test = x[None, :].expand(test_u0_rows.shape[0], -1)
        t_train = train_t_rows[:, None].expand(-1, x.numel())
        t_test = test_t_rows[:, None].expand(-1, x.numel())
        deeponet = DeepONet1D(num_points=x.numel(), latent=96, hidden=128).to(dtype=torch.float64)
        opt_deep = torch.optim.Adam(deeponet.parameters(), lr=1e-3)
        y_mean = train_y.mean()
        y_std = train_y.std().clamp_min(1e-12)
        train_y_norm = (train_y - y_mean) / y_std
        alpha_train = alpha_true.expand(train_u0_rows.shape[0])
        alpha_test = alpha_true.expand(test_u0_rows.shape[0])
        for _ in range(500):
            opt_deep.zero_grad()
            pred = deeponet(train_u0_rows, x_train, t_train, alpha_train)
            loss = torch.mean((pred - train_y_norm) ** 2)
            loss.backward()
            opt_deep.step()
        deep_train = (deeponet(train_u0_rows, x_train, t_train, alpha_train) * y_std + y_mean).detach()
        deep_test = (deeponet(test_u0_rows, x_test, t_test, alpha_test) * y_std + y_mean).detach()

        mlsl_test = torch.cat([layer(u0, test_times, alpha_true).detach() for u0 in test_u0])
        rows.extend(
            [
                {
                    "seed": seed,
                    "model": "MLSL_oracle",
                    "train_relative_error": 0.0,
                    "long_time_relative_error": rel(mlsl_test, test_y),
                },
                {
                    "seed": seed,
                    "model": "FNO1D_width48_modes16",
                    "train_relative_error": rel(fno_train, train_y),
                    "long_time_relative_error": rel(fno_test, test_y),
                },
                {
                    "seed": seed,
                    "model": "DeepONet_latent96_hidden128",
                    "train_relative_error": rel(deep_train, train_y),
                    "long_time_relative_error": rel(deep_test, test_y),
                },
            ]
        )
    summary = {}
    for model in sorted({str(r["model"]) for r in rows}):
        vals = [float(r["long_time_relative_error"]) for r in rows if r["model"] == model]
        m, s = mean_std(vals)
        summary[f"{model}_long_time_error_mean"] = m
        summary[f"{model}_long_time_error_std"] = s
    return rows, summary


def fpinn_repeated_baseline() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for seed in [0, 1, 2]:
        torch.manual_seed(4000 + seed)
        final_time = 1.0
        num_steps = 32
        mu = torch.tensor(1.4)
        u0 = torch.tensor(1.0)
        alpha_true = torch.tensor(0.65)
        grid = torch.linspace(0.0, final_time, num_steps + 1)
        clean = u0 * mittag_leffler_e(alpha_true, -mu * grid.pow(alpha_true), terms=140)
        sensor_idx = torch.linspace(0, num_steps, 7).round().long()
        observed = clean[sensor_idx] + 1e-4 * torch.std(clean) * torch.randn(sensor_idx.numel())

        model = ScalarTimeNet(hidden=48)
        raw_alpha = torch.nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.Adam(list(model.parameters()) + [raw_alpha], lr=2e-3)
        for _ in range(800):
            opt.zero_grad()
            alpha = 0.20 + 0.75 * torch.sigmoid(raw_alpha)
            pred = model(grid)
            caputo = l1_caputo_derivative_uniform(pred, alpha=alpha, final_time=final_time)
            residual = caputo + mu * pred[1:]
            loss = (
                10.0 * torch.mean((pred[sensor_idx] - observed) ** 2)
                + (pred[0] - u0) ** 2
                + 0.1 * torch.mean(residual**2)
            )
            loss.backward()
            opt.step()
        alpha_est = float((0.20 + 0.75 * torch.sigmoid(raw_alpha)).detach())
        pred = model(grid).detach()
        rows.append(
            {
                "seed": seed,
                "alpha_true": float(alpha_true),
                "alpha_est": alpha_est,
                "alpha_relative_error": abs(alpha_est - float(alpha_true)) / float(alpha_true),
                "solution_relative_error": rel(pred, clean),
            }
        )
    alpha_mean, alpha_std = mean_std([float(r["alpha_relative_error"]) for r in rows])
    sol_mean, sol_std = mean_std([float(r["solution_relative_error"]) for r in rows])
    return rows, {
        "fpinn_alpha_error_mean": alpha_mean,
        "fpinn_alpha_error_std": alpha_std,
        "fpinn_solution_error_mean": sol_mean,
        "fpinn_solution_error_std": sol_std,
    }


def forcing_quadrature_validation() -> tuple[list[dict[str, object]], dict[str, object]]:
    x, base = build_dirichlet_mlsl_1d(
        num_points=64,
        num_modes=16,
        config=MLSLConfig(terms=100),
    )
    u0 = 0.3 * torch.sin(torch.pi * x)
    alpha = torch.tensor(1.25, requires_grad=True)
    beta = torch.tensor(1.4, requires_grad=True)
    times = torch.linspace(0.0, 0.08, 6)

    def forcing_samples(q: int) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = (torch.arange(q, dtype=x.dtype) + 0.5) / q
        spatial = torch.sin(2.0 * torch.pi * x) + 0.1 * torch.sin(5.0 * torch.pi * x)
        values = torch.cos(0.3 * nodes[:, None]) * spatial[None, :]
        return values, nodes

    reference_values, reference_nodes = forcing_samples(192)
    reference_layer = ForcedMittagLefflerSpectralLayer(base, forcing_terms=120)
    reference = reference_layer(u0, times, alpha.detach(), reference_values, reference_nodes, beta=beta.detach()).detach()

    rows = []
    for q in [12, 24, 48, 96]:
        values, nodes = forcing_samples(q)
        layer = ForcedMittagLefflerSpectralLayer(base, forcing_terms=100)
        pred = layer(u0, times, alpha, values, nodes, beta=beta)
        loss = pred.square().mean()
        if alpha.grad is not None:
            alpha.grad.zero_()
        if beta.grad is not None:
            beta.grad.zero_()
        loss.backward(retain_graph=True)
        rows.append(
            {
                "quadrature_points": q,
                "relative_error_vs_q192": rel(pred.detach(), reference),
                "finite_output": bool(torch.isfinite(pred).all().item()),
                "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
            }
        )
    return rows, {
        "forcing_error_q12": float(rows[0]["relative_error_vs_q192"]),
        "forcing_error_q96": float(rows[-1]["relative_error_vs_q192"]),
        "forcing_convergence_ratio_q12_to_q96": float(rows[0]["relative_error_vs_q192"])
        / max(float(rows[-1]["relative_error_vs_q192"]), 1e-14),
    }


def proposition_evidence() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    alpha = torch.tensor(0.65)
    mu = torch.tensor(3.0)
    final_time = 0.5
    for num_steps in [100, 200, 400]:
        times = torch.linspace(0.0, final_time, num_steps + 1)
        start = time.perf_counter()
        _ = mittag_leffler_e(alpha, -mu * times.pow(alpha), terms=100, method="hybrid")
        mlsl_seconds = time.perf_counter() - start
        start = time.perf_counter()
        _ = l1_caputo_relaxation(torch.tensor(1.0), alpha=alpha, mu=mu, final_time=final_time, num_steps=num_steps)
        l1_seconds = time.perf_counter() - start
        rows.append(
            {
                "proposition": "history_free_time_query",
                "num_steps": num_steps,
                "mlsl_seconds": mlsl_seconds,
                "l1_seconds": l1_seconds,
                "speedup_l1_over_mlsl": l1_seconds / max(mlsl_seconds, 1e-14),
            }
        )
    return rows, {
        "proposition_history_free_speedup_at_400": rows[-1]["speedup_l1_over_mlsl"],
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    ensure_dirs()
    summary: dict[str, object] = {}

    experiments = [
        ("long_horizon_stability_accuracy.csv", long_horizon_stability_accuracy),
        ("multi_seed_inverse.csv", multi_seed_inverse),
        ("batch_and_device_profile.csv", batch_and_device_profile),
        ("stronger_operator_baselines.csv", stronger_operator_baselines),
        ("fpinn_repeated_baseline.csv", fpinn_repeated_baseline),
        ("forcing_quadrature_validation.csv", forcing_quadrature_validation),
        ("proposition_evidence.csv", proposition_evidence),
    ]
    for filename, fn in experiments:
        rows, partial = fn()
        write_csv(TABLES / filename, rows)
        summary.update(partial)

    (RESULTS / "paper_grade_extension_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

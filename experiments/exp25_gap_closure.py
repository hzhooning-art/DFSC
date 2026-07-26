"""Gap-closure experiments before paper writing."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import mpmath as mp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfsc import (
    DeepONet1D,
    FNO1D,
    MLSLConfig,
    MittagLefflerSpectralLayer,
    build_dirichlet_mlsl_1d,
    build_mixed_mlsl_1d,
    build_neumann_mlsl_1d,
    build_periodic_mlsl_1d,
    mittag_leffler_e_ab,
)


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


def make_random_initial_conditions(x: torch.Tensor, count: int, max_mode: int = 8) -> torch.Tensor:
    coeffs = torch.randn(count, max_mode) / torch.arange(1, max_mode + 1, dtype=x.dtype)
    modes = torch.stack([torch.sin((i + 1) * torch.pi * x) for i in range(max_mode)])
    fields = coeffs @ modes
    return fields / torch.linalg.norm(fields, dim=1, keepdim=True).clamp_min(1e-12)


def flatten_dataset(u0_batch: torch.Tensor, times: torch.Tensor, layer, alpha: torch.Tensor):
    u0_rows, t_rows, y_rows = [], [], []
    for u0 in u0_batch:
        y = layer(u0, times, alpha).detach()
        u0_rows.append(u0[None, :].expand(times.numel(), -1))
        t_rows.append(times)
        y_rows.append(y)
    return torch.cat(u0_rows), torch.cat(t_rows), torch.cat(y_rows)


def mp_mittag_leffler_ab(alpha: float, beta: float, z: float, terms: int = 260) -> float:
    mp.mp.dps = 80
    total = mp.mpf("0")
    z_mp = mp.mpf(str(z))
    for k in range(terms):
        term = z_mp**k / mp.gamma(alpha * k + beta)
        total += term
        if abs(term) < mp.mpf("1e-70"):
            break
    return float(total)


def two_parameter_reference_accuracy() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for alpha in [0.65, 0.85, 1.25]:
        for beta in [alpha, 1.0, 1.5]:
            for z in [-0.1, -1.0, -3.0, -6.0]:
                torch_value = mittag_leffler_e_ab(
                    torch.tensor(alpha),
                    torch.tensor(beta),
                    torch.tensor([z]),
                    terms=140,
                    method="hybrid",
                )[0].item()
                ref = mp_mittag_leffler_ab(alpha, beta, z)
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "z": z,
                        "hybrid_value": torch_value,
                        "mp_reference": ref,
                        "absolute_error": abs(torch_value - ref),
                        "relative_error": abs(torch_value - ref) / max(abs(ref), 1e-14),
                    }
                )
    return rows, {
        "eab_reference_max_relative_error": max(float(r["relative_error"]) for r in rows),
        "eab_reference_max_absolute_error": max(float(r["absolute_error"]) for r in rows),
    }


def mixed_precision_and_gpu_profile() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dtype in ([torch.float64, torch.float32] if device == "cpu" else [torch.float32, torch.float16]):
            cfg = MLSLConfig.stable(terms=120, dtype=dtype, device=device)
            x, layer = build_periodic_mlsl_1d(num_points=128, num_modes=31, config=cfg)
            layer = layer.to(device=device, dtype=dtype)
            x = x.to(device=device, dtype=dtype)
            u0 = (1.0 + 0.2 * torch.cos(2.0 * math.pi * x)).to(dtype=dtype, device=device)
            times = torch.linspace(0.0, 0.08, 10, dtype=dtype, device=device)
            alpha = torch.tensor(0.85, dtype=dtype, device=device)
            beta = torch.tensor(1.4, dtype=dtype, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = layer(u0[None, :].expand(512, -1), times, alpha, beta=beta)
            if device == "cuda":
                torch.cuda.synchronize()
            rows.append(
                {
                    "device": device,
                    "dtype": str(dtype).replace("torch.", ""),
                    "batch_size": 512,
                    "seconds": time.perf_counter() - start,
                    "finite_output": bool(torch.isfinite(out).all().item()),
                    "output_shape": str(tuple(out.shape)),
                }
            )
    return rows, {
        "gap_gpu_available": torch.cuda.is_available(),
        "mixed_precision_profiles": len(rows),
    }


def reaction_diffusion_family() -> tuple[list[dict[str, object]], dict[str, object]]:
    x, base = build_dirichlet_mlsl_1d(
        num_points=96,
        num_modes=18,
        config=MLSLConfig.stable(terms=120),
    )
    reaction = 1.25
    shifted = MittagLefflerSpectralLayer(
        base.eigenvalues + reaction,
        base.eigenvectors,
        terms=120,
        custom_backward=False,
        ml_method="hybrid",
    )
    u0 = torch.sin(torch.pi * x) + 0.2 * torch.sin(4.0 * torch.pi * x)
    rows = []
    for alpha_value in [0.65, 1.1, 1.6]:
        alpha = torch.tensor(alpha_value, requires_grad=True)
        beta = torch.tensor(1.5, requires_grad=True)
        out = shifted(u0, torch.linspace(0.0, 0.08, 8), alpha, beta=beta)
        loss = out.square().mean()
        loss.backward()
        rows.append(
            {
                "pde_family": "linear_reaction_fractional_diffusion",
                "reaction": reaction,
                "alpha": alpha_value,
                "beta": 1.5,
                "finite_output": bool(torch.isfinite(out).all().item()),
                "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                "loss": float(loss.detach()),
            }
        )
    return rows, {
        "reaction_diffusion_pass_rate": sum(
            1 for r in rows if r["finite_output"] and r["finite_alpha_grad"] and r["finite_beta_grad"]
        )
        / len(rows)
    }


def boundary_extended_matrix() -> tuple[list[dict[str, object]], dict[str, object]]:
    builders = {
        "dirichlet": lambda: build_dirichlet_mlsl_1d(num_points=72, num_modes=14, config=MLSLConfig.stable()),
        "neumann": lambda: build_neumann_mlsl_1d(num_points=72, num_modes=14, config=MLSLConfig.stable()),
        "periodic": lambda: build_periodic_mlsl_1d(num_points=72, num_modes=15, config=MLSLConfig.stable()),
        "mixed_dn": lambda: build_mixed_mlsl_1d(num_points=72, num_modes=14, boundary="dn", config=MLSLConfig.stable()),
        "mixed_nd": lambda: build_mixed_mlsl_1d(num_points=72, num_modes=14, boundary="nd", config=MLSLConfig.stable()),
    }
    rows = []
    for name, build in builders.items():
        x, layer = build()
        u0 = torch.ones_like(x) if name in {"neumann", "periodic"} else layer.eigenvectors[:, 0].to(dtype=x.dtype)
        alpha = torch.tensor(0.9, requires_grad=True)
        beta = torch.tensor(1.6, requires_grad=True)
        out = layer(u0, torch.linspace(0.0, 0.05, 6), alpha, beta=beta)
        out.square().mean().backward()
        rows.append(
            {
                "boundary": name,
                "finite_output": bool(torch.isfinite(out).all().item()),
                "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
            }
        )
    return rows, {
        "extended_boundary_pass_rate": sum(
            1 for r in rows if r["finite_output"] and r["finite_alpha_grad"] and r["finite_beta_grad"]
        )
        / len(rows)
    }


def larger_neural_baseline_multiseed() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for seed in [0, 1, 2]:
        torch.manual_seed(5100 + seed)
        x, layer = build_dirichlet_mlsl_1d(
            num_points=64,
            num_modes=24,
            config=MLSLConfig.stable(terms=120),
        )
        alpha_true = torch.tensor(1.35)
        train_times = torch.linspace(0.0, 0.05, 7)
        test_times = torch.linspace(0.12, 0.28, 6)
        train_u0 = make_random_initial_conditions(x, 36)
        test_u0 = make_random_initial_conditions(x, 12)
        train_u0_rows, train_t_rows, train_y = flatten_dataset(train_u0, train_times, layer, alpha_true)
        test_u0_rows, test_t_rows, test_y = flatten_dataset(test_u0, test_times, layer, alpha_true)

        fno = FNO1D(modes=16, width=56, layers=4).to(dtype=torch.float64)
        opt = torch.optim.Adam(fno.parameters(), lr=2e-3)
        for _ in range(260):
            opt.zero_grad()
            loss = torch.mean((fno(train_u0_rows, train_t_rows) - train_y) ** 2)
            loss.backward()
            opt.step()
        fno_test = fno(test_u0_rows, test_t_rows).detach()

        x_test = x[None, :].expand(test_u0_rows.shape[0], -1)
        t_test = test_t_rows[:, None].expand(-1, x.numel())
        x_train = x[None, :].expand(train_u0_rows.shape[0], -1)
        t_train = train_t_rows[:, None].expand(-1, x.numel())
        deeponet = DeepONet1D(num_points=x.numel(), latent=96, hidden=128).to(dtype=torch.float64)
        opt_d = torch.optim.Adam(deeponet.parameters(), lr=1e-3)
        y_mean = train_y.mean()
        y_std = train_y.std().clamp_min(1e-12)
        y_norm = (train_y - y_mean) / y_std
        alpha_train = alpha_true.expand(train_u0_rows.shape[0])
        alpha_test = alpha_true.expand(test_u0_rows.shape[0])
        for _ in range(320):
            opt_d.zero_grad()
            pred = deeponet(train_u0_rows, x_train, t_train, alpha_train)
            loss = torch.mean((pred - y_norm) ** 2)
            loss.backward()
            opt_d.step()
        deep_test = (deeponet(test_u0_rows, x_test, t_test, alpha_test) * y_std + y_mean).detach()
        rows.extend(
            [
                {"seed": seed, "model": "FNO1D_width56_modes16", "long_time_relative_error": rel(fno_test, test_y)},
                {"seed": seed, "model": "DeepONet_latent96_hidden128", "long_time_relative_error": rel(deep_test, test_y)},
            ]
        )
    summary = {}
    for model in sorted({r["model"] for r in rows}):
        values = [float(r["long_time_relative_error"]) for r in rows if r["model"] == model]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1))
        summary[f"{model}_gap_multiseed_mean"] = mean
        summary[f"{model}_gap_multiseed_std"] = std
    return rows, summary


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("eab_reference_accuracy.csv", two_parameter_reference_accuracy),
        ("mixed_precision_gpu_profile.csv", mixed_precision_and_gpu_profile),
        ("reaction_diffusion_family.csv", reaction_diffusion_family),
        ("boundary_extended_matrix.csv", boundary_extended_matrix),
        ("larger_neural_baseline_multiseed.csv", larger_neural_baseline_multiseed),
    ]
    summary: dict[str, object] = {}
    for filename, fn in jobs:
        rows, part = fn()
        write_csv(TABLES / filename, rows)
        summary.update(part)
        print("saved", filename)
    (RESULTS / "gap_closure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""GPU validation and profiling for the MLSL primitive."""

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

from dfsc import MLSLConfig, build_dirichlet_mlsl_1d, build_periodic_mlsl_1d


RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-14)).item()


def make_batch(x: torch.Tensor, batch_size: int, max_mode: int = 6) -> torch.Tensor:
    coeffs = torch.randn(batch_size, max_mode, dtype=x.dtype, device=x.device)
    coeffs = coeffs / torch.arange(1, max_mode + 1, dtype=x.dtype, device=x.device)
    modes = torch.stack([torch.sin((i + 1) * torch.pi * x) for i in range(max_mode)])
    fields = coeffs @ modes
    return fields / torch.linalg.norm(fields, dim=1, keepdim=True).clamp_min(1e-12)


def cpu_gpu_consistency() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for dtype in [torch.float64, torch.float32]:
        x_cpu, layer_cpu = build_dirichlet_mlsl_1d(
            num_points=128,
            num_modes=24,
            config=MLSLConfig.stable(terms=120, dtype=dtype, device="cpu"),
        )
        x_gpu, layer_gpu = build_dirichlet_mlsl_1d(
            num_points=128,
            num_modes=24,
            config=MLSLConfig.stable(terms=120, dtype=dtype, device="cuda"),
        )
        u0_cpu = torch.sin(torch.pi * x_cpu) + 0.2 * torch.sin(4.0 * torch.pi * x_cpu)
        u0_gpu = u0_cpu.to("cuda")
        times_cpu = torch.linspace(0.0, 0.08, 9, dtype=dtype)
        times_gpu = times_cpu.to("cuda")
        alpha_cpu = torch.tensor(0.85, dtype=dtype, requires_grad=True)
        beta_cpu = torch.tensor(1.45, dtype=dtype, requires_grad=True)
        alpha_gpu = torch.tensor(0.85, dtype=dtype, device="cuda", requires_grad=True)
        beta_gpu = torch.tensor(1.45, dtype=dtype, device="cuda", requires_grad=True)

        out_cpu = layer_cpu(u0_cpu, times_cpu, alpha_cpu, beta=beta_cpu)
        loss_cpu = out_cpu.square().mean()
        loss_cpu.backward()
        out_gpu = layer_gpu(u0_gpu, times_gpu, alpha_gpu, beta=beta_gpu)
        loss_gpu = out_gpu.square().mean()
        loss_gpu.backward()
        torch.cuda.synchronize()

        rows.append(
            {
                "check": "cpu_gpu_consistency",
                "dtype": str(dtype).replace("torch.", ""),
                "output_relative_error": rel_error(out_gpu.detach().cpu(), out_cpu.detach()),
                "alpha_grad_relative_error": abs(alpha_gpu.grad.detach().cpu().item() - alpha_cpu.grad.item())
                / max(abs(alpha_cpu.grad.item()), 1e-14),
                "beta_grad_relative_error": abs(beta_gpu.grad.detach().cpu().item() - beta_cpu.grad.item())
                / max(abs(beta_cpu.grad.item()), 1e-14),
                "finite_gpu_output": bool(torch.isfinite(out_gpu).all().item()),
                "finite_gpu_alpha_grad": bool(torch.isfinite(alpha_gpu.grad).item()),
                "finite_gpu_beta_grad": bool(torch.isfinite(beta_gpu.grad).item()),
            }
        )
    return rows, {
        "gpu_consistency_max_output_relative_error": max(float(r["output_relative_error"]) for r in rows),
        "gpu_consistency_max_alpha_grad_relative_error": max(float(r["alpha_grad_relative_error"]) for r in rows),
        "gpu_consistency_max_beta_grad_relative_error": max(float(r["beta_grad_relative_error"]) for r in rows),
    }


def gpu_batch_profile() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for dtype in [torch.float64, torch.float32]:
        x, layer = build_periodic_mlsl_1d(
            num_points=256,
            num_modes=41,
            config=MLSLConfig.stable(terms=120, dtype=dtype, device="cuda"),
        )
        times = torch.linspace(0.0, 0.08, 12, dtype=dtype, device="cuda")
        alpha = torch.tensor(0.9, dtype=dtype, device="cuda", requires_grad=True)
        beta = torch.tensor(1.4, dtype=dtype, device="cuda", requires_grad=True)
        for batch_size in [1, 16, 64, 256, 1024]:
            u0 = make_batch(x, batch_size)
            for _ in range(3):
                layer(u0, times, alpha, beta=beta)
            torch.cuda.synchronize()
            timings = []
            for _ in range(8):
                start = time.perf_counter()
                out = layer(u0, times, alpha, beta=beta)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            loss = out.square().mean()
            if alpha.grad is not None:
                alpha.grad.zero_()
            if beta.grad is not None:
                beta.grad.zero_()
            loss.backward()
            torch.cuda.synchronize()
            rows.append(
                {
                    "check": "gpu_batch_profile",
                    "dtype": str(dtype).replace("torch.", ""),
                    "batch_size": batch_size,
                    "seconds_min": min(timings),
                    "seconds_per_sample": min(timings) / batch_size,
                    "finite_output": bool(torch.isfinite(out).all().item()),
                    "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
                    "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
                    "output_shape": str(tuple(out.shape)),
                }
            )
    return rows, {
        "gpu_float64_seconds_at_batch1024": [
            float(r["seconds_min"]) for r in rows if r["dtype"] == "float64" and r["batch_size"] == 1024
        ][0],
        "gpu_float32_seconds_at_batch1024": [
            float(r["seconds_min"]) for r in rows if r["dtype"] == "float32" and r["batch_size"] == 1024
        ][0],
    }


def half_precision_probe() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for dtype in [torch.float16, torch.bfloat16]:
        try:
            x, layer = build_dirichlet_mlsl_1d(
                num_points=64,
                num_modes=12,
                config=MLSLConfig.stable(terms=80, dtype=dtype, device="cuda"),
            )
            u0 = torch.sin(torch.pi * x)
            out = layer(
                u0,
                torch.linspace(0.0, 0.03, 5, dtype=dtype, device="cuda"),
                torch.tensor(0.85, dtype=dtype, device="cuda"),
                beta=torch.tensor(1.3, dtype=dtype, device="cuda"),
            )
            torch.cuda.synchronize()
            rows.append(
                {
                    "check": "half_precision_probe",
                    "dtype": str(dtype).replace("torch.", ""),
                    "supported": True,
                    "finite_output": bool(torch.isfinite(out).all().item()),
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - record capability boundary.
            rows.append(
                {
                    "check": "half_precision_probe",
                    "dtype": str(dtype).replace("torch.", ""),
                    "supported": False,
                    "finite_output": False,
                    "message": type(exc).__name__ + ": " + str(exc)[:180],
                }
            )
    return rows, {
        "half_precision_supported_count": sum(1 for r in rows if r["supported"]),
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        summary = {
            "cuda_available": False,
            "message": "CUDA is not available in the active PyTorch environment.",
        }
        (RESULTS / "gpu_validation_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return

    rows = []
    summary: dict[str, object] = {
        "cuda_available": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }
    for filename, fn in [
        ("gpu_cpu_consistency.csv", cpu_gpu_consistency),
        ("gpu_batch_profile.csv", gpu_batch_profile),
        ("gpu_half_precision_probe.csv", half_precision_probe),
    ]:
        part_rows, part_summary = fn()
        write_csv(TABLES / filename, part_rows)
        rows.extend(part_rows)
        summary.update(part_summary)

    summary["gpu_all_finite_checks"] = all(
        bool(r.get("finite_output", True)) for r in rows if "finite_output" in r
    )
    (RESULTS / "gpu_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

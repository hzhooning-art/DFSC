"""Accuracy, gradient, scaling, and CUDA validation for FFT Caputo-L1 history."""

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

import dfsc


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return float((torch.linalg.vector_norm(actual - reference) / denominator).detach().cpu())


def timed(function, repeats: int = 5) -> tuple[torch.Tensor, float]:
    durations = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - started)
    assert result is not None
    return result, statistics.median(durations)


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = (
        path,
        path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}"),
        ROOT / f"{path.stem}_{int(time.time())}{path.suffix}",
    )
    for candidate in candidates:
        try:
            handle = candidate.open("w", newline="", encoding="utf-8")
            path = candidate
            break
        except PermissionError:
            continue
    else:
        raise PermissionError(f"could not write {path}")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    torch.manual_seed(38)
    torch.set_default_dtype(torch.float64)
    alpha = torch.tensor(0.68)

    timing_rows: list[dict[str, object]] = []
    for steps in (128, 256, 512, 1024, 2048):
        values = torch.sin(torch.linspace(0.0, 4.0, steps + 1))
        direct, direct_seconds = timed(
            lambda: dfsc.caputo_l1_derivative_direct(values, alpha=alpha, final_time=1.0)[0]
        )
        fft, fft_seconds = timed(
            lambda: dfsc.caputo_l1_derivative_fft(values, alpha=alpha, final_time=1.0)[0]
        )
        timing_rows.append(
            {
                "num_steps": steps,
                "direct_seconds": direct_seconds,
                "fft_seconds": fft_seconds,
                "measured_speedup": direct_seconds / fft_seconds,
                "relative_error_fft_vs_direct": relative_error(fft, direct),
            }
        )

    large_rows: list[dict[str, object]] = []
    for steps in (8192, 32768, 65536):
        values = torch.sin(torch.linspace(0.0, 8.0, steps + 1))
        fft, fft_seconds = timed(
            lambda: dfsc.caputo_l1_derivative_fft(values, alpha=alpha, final_time=2.0)[0],
            repeats=3,
        )
        large_rows.append(
            {
                "num_steps": steps,
                "fft_seconds": fft_seconds,
                "finite": bool(torch.isfinite(fft).all()),
            }
        )

    convergence_rows: list[dict[str, object]] = []
    power = torch.tensor(2.0)
    for steps in (128, 512, 2048):
        times = torch.linspace(0.0, 1.0, steps + 1)
        numerical, _ = dfsc.caputo_l1_derivative_fft(
            times.pow(power), alpha=alpha, final_time=1.0
        )
        exact = (
            torch.exp(torch.lgamma(power + 1.0) - torch.lgamma(power + 1.0 - alpha))
            * times[1:].pow(power - alpha)
        )
        convergence_rows.append(
            {
                "num_steps": steps,
                "relative_error": relative_error(numerical, exact),
            }
        )

    values_direct = torch.randn(2, 257, 3, requires_grad=True)
    values_fft = values_direct.detach().clone().requires_grad_(True)
    alpha_direct = torch.tensor(0.68, requires_grad=True)
    alpha_fft = alpha_direct.detach().clone().requires_grad_(True)
    direct, _ = dfsc.caputo_l1_derivative_direct(values_direct, alpha=alpha_direct, final_time=1.0)
    fft, _ = dfsc.caputo_l1_derivative_fft(values_fft, alpha=alpha_fft, final_time=1.0)
    direct.square().mean().backward()
    fft.square().mean().backward()

    gpu = {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "relative_error_vs_cpu": None,
        "alpha_gradient_finite": None,
        "trajectory_gradient_finite": None,
    }
    if torch.cuda.is_available():
        gpu_values = torch.randn(4, 8193, 3, dtype=torch.float64)
        gpu_alpha = torch.tensor(0.68, dtype=torch.float64, device="cuda", requires_grad=True)
        gpu_input = gpu_values.to("cuda").requires_grad_(True)
        gpu_derivative, _ = dfsc.caputo_l1_derivative_fft(
            gpu_input, alpha=gpu_alpha, final_time=2.0
        )
        cpu_derivative, _ = dfsc.caputo_l1_derivative_fft(
            gpu_values, alpha=alpha, final_time=2.0
        )
        gpu_derivative.square().mean().backward()
        gpu["relative_error_vs_cpu"] = relative_error(gpu_derivative.cpu(), cpu_derivative)
        gpu["alpha_gradient_finite"] = bool(torch.isfinite(gpu_alpha.grad))
        gpu["trajectory_gradient_finite"] = bool(torch.isfinite(gpu_input.grad).all())

    table_path = write_csv(ROOT / "results" / "tables" / "fast_history_timing.csv", timing_rows)
    convergence_path = write_csv(
        ROOT / "results" / "tables" / "fast_history_convergence.csv", convergence_rows
    )
    large_path = write_csv(ROOT / "results" / "tables" / "fast_history_large_fft.csv", large_rows)
    summary = {
        "max_fft_vs_direct_relative_error": max(row["relative_error_fft_vs_direct"] for row in timing_rows),
        "alpha_gradient_relative_error": relative_error(alpha_fft.grad, alpha_direct.grad),
        "trajectory_gradient_relative_error": relative_error(values_fft.grad, values_direct.grad),
        "analytic_error_128": convergence_rows[0]["relative_error"],
        "analytic_error_2048": convergence_rows[-1]["relative_error"],
        "analytic_error_reduction": convergence_rows[0]["relative_error"] / convergence_rows[-1]["relative_error"],
        "measured_speedup_2048": timing_rows[-1]["measured_speedup"],
        "largest_fft_steps": large_rows[-1]["num_steps"],
        "largest_fft_seconds": large_rows[-1]["fft_seconds"],
        "largest_fft_finite": large_rows[-1]["finite"],
        "gpu": gpu,
        "scope": "offline full-trajectory Caputo-L1 derivative; not an online implicit time stepper",
    }
    summary_path = ROOT / "results" / "fast_history_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except PermissionError:
        summary_path = summary_path.with_name(f"{summary_path.stem}_{int(time.time())}.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary.update(
        {
            "timing_table": str(table_path),
            "convergence_table": str(convergence_path),
            "large_fft_table": str(large_path),
            "summary_path": str(summary_path),
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Float32/float64 accuracy and hardware trade-off for two primitives.

The matrix-action reference uses SciPy expm and a finite-difference directional
derivative.  The periodic-heat reference uses an analytic mode expansion.  The
experiment therefore does not certify either backend against its own autodiff
trace.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "P4" / "results" / "p4_precision_tradeoff.json"
DTYPES = (torch.float32, torch.float64)
WARMUP = 10
REPEATS = 50


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(callable_, device: torch.device) -> tuple[float, int | None]:
    for _ in range(WARMUP):
        callable_()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    start = time.perf_counter()
    for _ in range(REPEATS):
        callable_()
    synchronize(device)
    latency_ms = 1000.0 * (time.perf_counter() - start) / REPEATS
    peak = None
    if device.type == "cuda":
        peak = int(max(0, torch.cuda.max_memory_allocated(device) - baseline))
    return latency_ms, peak


def matrix_action(inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    matrix = parameters.reshape(2, 2)
    exponent = torch.matrix_exp(inputs[:, :1, None] * matrix)
    return torch.bmm(exponent, inputs[:, 1:, None]).squeeze(-1)


def scipy_matrix_action(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    matrix = parameters.reshape(2, 2)
    return np.stack([expm(float(row[0]) * matrix) @ row[1:] for row in inputs])


def matrix_row(dtype: torch.dtype, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(76101)
    times = torch.linspace(0.02, 2.0, 1024, dtype=torch.float64)
    vectors = torch.randn((1024, 2), generator=generator, dtype=torch.float64)
    inputs64 = torch.cat((times[:, None], vectors), dim=1)
    parameters64 = torch.tensor([-0.70, 0.20, -0.10, -0.40], dtype=torch.float64)
    direction64 = torch.tensor([0.30, -0.20, 0.40, -0.50], dtype=torch.float64)

    reference = scipy_matrix_action(inputs64.numpy(), parameters64.numpy())
    epsilon = 1.0e-6
    plus = scipy_matrix_action(inputs64.numpy(), (parameters64 + epsilon * direction64).numpy())
    minus = scipy_matrix_action(inputs64.numpy(), (parameters64 - epsilon * direction64).numpy())
    reference_directional = float(((plus - minus) / (2.0 * epsilon)).mean())

    inputs = inputs64.to(device=device, dtype=dtype)
    parameters = parameters64.to(device=device, dtype=dtype).requires_grad_(True)
    direction = direction64.to(device=device, dtype=dtype)
    prediction = matrix_action(inputs, parameters)
    scalar = prediction.mean()
    gradient = torch.autograd.grad(scalar, parameters)[0]
    directional = float(torch.dot(gradient, direction).detach().cpu())
    value_error = float(np.max(np.abs(prediction.detach().cpu().double().numpy() - reference)))
    gradient_abs_error = abs(directional - reference_directional)
    gradient_rel_error = gradient_abs_error / max(abs(reference_directional), 1.0e-15)
    latency_ms, peak = timed(lambda: matrix_action(inputs, parameters.detach()), device)
    return {
        "primitive": "matrix_exponential_action",
        "dtype": str(dtype).replace("torch.", ""),
        "batch": int(inputs.shape[0]),
        "max_abs_value_error": value_error,
        "directional_gradient": directional,
        "reference_directional_gradient": reference_directional,
        "directional_gradient_abs_error": gradient_abs_error,
        "directional_gradient_relative_error": gradient_rel_error,
        "mean_forward_ms": latency_ms,
        "peak_incremental_cuda_bytes": peak,
        "finite": bool(torch.isfinite(prediction).all() and torch.isfinite(gradient).all()),
    }


MODES = ((1, 0), (0, 2), (3, -2), (4, 3), (5, -1))


def analytic_heat(n: int, coeff: torch.Tensor, kappa: torch.Tensor, horizon: float):
    axis = 2.0 * math.pi * torch.arange(n, device=coeff.device, dtype=coeff.dtype) / n
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    field = torch.zeros((n, n), device=coeff.device, dtype=coeff.dtype)
    derivative = torch.zeros_like(field)
    for index, (kx, ky) in enumerate(MODES):
        wave_sq = float(kx * kx + ky * ky)
        phase = kx * x + ky * y + 0.17 * index
        basis = torch.sin(phase) if index % 2 == 0 else torch.cos(phase)
        decay = torch.exp(-kappa * wave_sq * horizon)
        term = coeff[index] * decay * basis
        field = field + term
        derivative = derivative - horizon * wave_sq * term
    return field, derivative


def spectral_heat(initial: torch.Tensor, kappa: torch.Tensor, horizon: float) -> torch.Tensor:
    n = initial.shape[-1]
    frequencies = torch.fft.fftfreq(n, d=1.0 / n, device=initial.device, dtype=initial.dtype)
    kx, ky = torch.meshgrid(frequencies, frequencies, indexing="ij")
    multiplier = torch.exp(-kappa * (kx.square() + ky.square()) * horizon)
    return torch.fft.ifft2(torch.fft.fft2(initial) * multiplier).real


def heat_row(dtype: torch.dtype, device: torch.device) -> dict:
    n = 256
    horizon = 1.0
    generator = torch.Generator(device="cpu").manual_seed(76102)
    coeff64 = 0.25 + 0.75 * torch.rand(len(MODES), generator=generator, dtype=torch.float64)
    kappa64 = torch.tensor(0.08, dtype=torch.float64)
    initial64, _ = analytic_heat(n, coeff64, kappa64, 0.0)
    reference, reference_derivative = analytic_heat(n, coeff64, kappa64, horizon)
    reference_gradient = float((2.0 * reference * reference_derivative).mean())

    initial = initial64.to(device=device, dtype=dtype)
    kappa = kappa64.to(device=device, dtype=dtype).requires_grad_(True)
    prediction = spectral_heat(initial, kappa, horizon)
    gradient = torch.autograd.grad(prediction.square().mean(), kappa)[0]
    value_error = float((prediction.detach().cpu().double() - reference).abs().max())
    gradient_value = float(gradient.detach().cpu())
    gradient_abs_error = abs(gradient_value - reference_gradient)
    gradient_rel_error = gradient_abs_error / max(abs(reference_gradient), 1.0e-15)
    latency_ms, peak = timed(lambda: spectral_heat(initial, kappa.detach(), horizon), device)
    return {
        "primitive": "periodic_heat_2d_spectral",
        "dtype": str(dtype).replace("torch.", ""),
        "grid": n,
        "state_dimension": n * n,
        "max_abs_value_error": value_error,
        "energy_gradient": gradient_value,
        "reference_energy_gradient": reference_gradient,
        "energy_gradient_abs_error": gradient_abs_error,
        "energy_gradient_relative_error": gradient_rel_error,
        "mean_forward_ms": latency_ms,
        "peak_incremental_cuda_bytes": peak,
        "finite": bool(torch.isfinite(prediction).all() and torch.isfinite(gradient)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dtype in DTYPES:
        rows.append(matrix_row(dtype, device))
        rows.append(heat_row(dtype, device))
    result = {
        "schema": "DFSC-P4-Precision-Tradeoff-v1",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "rows": rows,
        "interpretation": (
            "Dtype-specific errors and profiles are valid only for the declared workloads. "
            "The results calibrate gate tolerances; they do not define universal float32 or float64 thresholds."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

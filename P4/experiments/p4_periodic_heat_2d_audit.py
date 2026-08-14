"""Direct 2D PDE audit for the differentiable-primitive reliability protocol.

The experiment evaluates a Fourier heat-propagation primitive on periodic
two-dimensional grids.  A finite trigonometric mode expansion provides a
closed-form reference that is independent of the FFT implementation.  The
audit records value error, an autodiff parameter-gradient check, long-horizon
behavior, execution time, and peak CUDA memory across grid refinements.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "P4" / "results" / "p4_periodic_heat_2d_audit.json"
GRIDS = (16, 32, 64, 128, 256)
HORIZONS = (0.1, 0.5, 1.0)
SEEDS = (74101, 74102, 74103, 74104, 74105)
KAPPA = 0.08
MODES = ((1, 0), (0, 2), (3, -2), (4, 3), (5, -1))


def coefficients(seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return 0.25 + 0.75 * torch.rand(len(MODES), generator=generator, device=device, dtype=dtype)


def coordinates(n: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    axis = 2.0 * math.pi * torch.arange(n, device=device, dtype=dtype) / n
    return torch.meshgrid(axis, axis, indexing="ij")


def analytic_field(
    n: int,
    coeff: torch.Tensor,
    kappa: torch.Tensor,
    horizon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = coordinates(n, coeff.device, coeff.dtype)
    field = torch.zeros((n, n), device=coeff.device, dtype=coeff.dtype)
    derivative = torch.zeros_like(field)
    for index, (kx, ky) in enumerate(MODES):
        wave_number_sq = float(kx * kx + ky * ky)
        phase = kx * x + ky * y + 0.17 * index
        basis = torch.sin(phase) if index % 2 == 0 else torch.cos(phase)
        decay = torch.exp(-kappa * wave_number_sq * horizon)
        term = coeff[index] * decay * basis
        field = field + term
        derivative = derivative - horizon * wave_number_sq * term
    return field, derivative


def spectral_heat_primitive(initial: torch.Tensor, kappa: torch.Tensor, horizon: float) -> torch.Tensor:
    n = initial.shape[-1]
    frequencies = torch.fft.fftfreq(n, d=1.0 / n, device=initial.device, dtype=initial.dtype)
    kx, ky = torch.meshgrid(frequencies, frequencies, indexing="ij")
    multiplier = torch.exp(-kappa * (kx.square() + ky.square()) * horizon)
    return torch.fft.ifft2(torch.fft.fft2(initial) * multiplier).real


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(initial: torch.Tensor, kappa: torch.Tensor, horizon: float) -> tuple[float, int | None]:
    device = initial.device
    for _ in range(5):
        spectral_heat_primitive(initial, kappa, horizon)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    start = time.perf_counter()
    for _ in range(20):
        spectral_heat_primitive(initial, kappa, horizon)
    synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / 20.0
    peak = None
    if device.type == "cuda":
        peak = int(max(0, torch.cuda.max_memory_allocated(device) - baseline))
    return elapsed_ms, peak


def main() -> None:
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    rows = []
    for n in GRIDS:
        for seed in SEEDS:
            coeff = coefficients(seed, device, dtype)
            kappa = torch.tensor(KAPPA, device=device, dtype=dtype, requires_grad=True)
            initial, _ = analytic_field(n, coeff, kappa.detach(), 0.0)
            for horizon in HORIZONS:
                prediction = spectral_heat_primitive(initial, kappa, horizon)
                reference, reference_derivative = analytic_field(n, coeff, kappa.detach(), horizon)
                value_error = float((prediction - reference).abs().max().detach().cpu())

                # The scalar energy is a reproducible directional functional of
                # the field; its derivative avoids materializing a full Jacobian.
                energy = prediction.square().mean()
                gradient = torch.autograd.grad(energy, kappa, retain_graph=False)[0]
                reference_gradient = (2.0 * reference * reference_derivative).mean()
                gradient_abs_error = float((gradient - reference_gradient).abs().detach().cpu())
                gradient_rel_error = gradient_abs_error / max(
                    float(reference_gradient.abs().detach().cpu()), 1.0e-15
                )

                elapsed_ms, peak_bytes = timed_forward(initial, kappa.detach(), horizon)
                rows.append(
                    {
                        "grid": n,
                        "state_dimension": n * n,
                        "seed": seed,
                        "horizon": horizon,
                        "max_abs_value_error": value_error,
                        "energy_gradient": float(gradient.detach().cpu()),
                        "reference_energy_gradient": float(reference_gradient.detach().cpu()),
                        "energy_gradient_abs_error": gradient_abs_error,
                        "energy_gradient_relative_error": gradient_rel_error,
                        "mean_forward_ms": elapsed_ms,
                        "peak_incremental_cuda_bytes": peak_bytes,
                        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(gradient).item()),
                    }
                )

    result = {
        "schema": "DFSC-P4-Periodic-Heat-2D-Audit-v1",
        "equation": "u_t = kappa * Laplacian(u) on [0, 2*pi]^2 with periodic boundaries",
        "primitive": "batched-compatible torch.fft spectral propagator",
        "reference": "closed-form finite trigonometric mode expansion",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "kappa": KAPPA,
        "grids": list(GRIDS),
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "timing_warmup": 5,
        "timing_repeats": 20,
        "rows": rows,
        "scope": (
            "Direct validation for a periodic linear 2D heat equation with a spectral diagonalization; "
            "not validation for nonlinear, stiff multiphysics, nonperiodic, or unstructured-grid PDEs."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()

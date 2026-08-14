"""Standardized batch throughput and GPU-memory profile for generic backends."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_generic_matrix_exp_validation import MatrixExponentialAction  # noqa: E402
from p4_generic_ode_step_validation import RK4LinearStep  # noqa: E402
from p4_nonlinear_ode_step_validation import LogisticRK4Step  # noqa: E402


def profile(name, backend, input_builder, parameter_builder, device, dtype, batches):
    rows = []
    for batch in batches:
        inputs = input_builder(batch, device, dtype)
        parameters = parameter_builder(device, dtype)
        for _ in range(10):
            backend(inputs, parameters)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            outputs = backend(inputs, parameters)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "backend": name,
                "batch": batch,
                "dtype": str(dtype),
                "device": str(device),
                "iterations": iterations,
                "mean_latency_ms": 1000.0 * elapsed / iterations,
                "samples_per_second": batch * iterations / elapsed,
                "output_shape": list(outputs.shape),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "outputs_finite": bool(torch.isfinite(outputs).all().item()),
            }
        )
    return rows


def matrix_inputs(batch, device, dtype):
    return torch.cat((torch.rand(batch, 1, dtype=dtype, device=device), torch.randn(batch, 2, dtype=dtype, device=device)), dim=1)


def matrix_params(device, dtype):
    return torch.tensor([-0.70, 0.20, -0.10, -0.40], dtype=dtype, device=device)


def logistic_inputs(batch, device, dtype):
    return torch.cat((0.01 + 0.24 * torch.rand(batch, 1, dtype=dtype, device=device), 0.1 + 0.8 * torch.rand(batch, 1, dtype=dtype, device=device)), dim=1)


def logistic_params(device, dtype):
    return torch.tensor([0.8, 2.0], dtype=dtype, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    batches = [1, 64, 256, 1024]
    rows = []
    rows.extend(profile("matrix_exponential_action", MatrixExponentialAction(), matrix_inputs, matrix_params, device, dtype, batches))
    rows.extend(profile("rk4_linear_ode_step", RK4LinearStep(), matrix_inputs, matrix_params, device, dtype, batches))
    rows.extend(profile("logistic_rk4_step", LogisticRK4Step(), logistic_inputs, logistic_params, device, dtype, batches))
    result = {
        "schema": "DFSC-Primitive-Profile-v1",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_sizes": batches,
        "rows": rows,
        "timing_rule": "10 warmup calls, 50 measured calls, device synchronization before timing readout",
        "interpretation": "comparable runtime/memory profile for three generic backends; not a cross-domain speed ranking",
    }
    out = ROOT / "P4" / "results" / "p4_primitive_profile.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()

"""Diagnostics and capability reporting for dfsc."""

from __future__ import annotations

import platform
from typing import Any

import torch

from .factory import MLSLConfig, build_mlsl


def environment_report() -> dict[str, Any]:
    """Return a machine-readable report of the active runtime."""

    cuda_available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_built": torch.backends.cuda.is_built(),
        "default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
    }
    if cuda_available:
        report.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_capability": torch.cuda.get_device_capability(0),
            }
        )
    return report


def smoke_test(*, device: str = "cpu", dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    """Run a small forward/backward test for a selected runtime."""

    config = MLSLConfig.stable(terms=80, dtype=dtype, device=device)
    x, layer = build_mlsl(
        dimension=1,
        boundary="dirichlet",
        num_points=48,
        num_modes=10,
        config=config,
    )
    u0 = torch.sin(torch.pi * x)
    times = torch.linspace(0.0, 0.03, 5, dtype=dtype, device=device)
    alpha = torch.tensor(0.85, dtype=dtype, device=device, requires_grad=True)
    beta = torch.tensor(1.4, dtype=dtype, device=device, requires_grad=True)
    out = layer(u0, times, alpha, beta=beta)
    loss = out.square().mean()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "output_shape": tuple(out.shape),
        "finite_output": bool(torch.isfinite(out).all().item()),
        "finite_alpha_grad": bool(torch.isfinite(alpha.grad).item()),
        "finite_beta_grad": bool(torch.isfinite(beta.grad).item()),
        "loss": float(loss.detach().cpu()),
    }


def capability_report() -> dict[str, Any]:
    """Return environment and smoke-test reports for available runtimes."""

    report = {"environment": environment_report(), "smoke_tests": []}
    smoke_tests = [smoke_test(device="cpu", dtype=torch.float64)]
    if torch.cuda.is_available():
        smoke_tests.append(smoke_test(device="cuda", dtype=torch.float64))
        smoke_tests.append(smoke_test(device="cuda", dtype=torch.float32))
    report["smoke_tests"] = smoke_tests
    report["all_smoke_tests_passed"] = all(
        row["finite_output"] and row["finite_alpha_grad"] and row["finite_beta_grad"]
        for row in smoke_tests
    )
    return report

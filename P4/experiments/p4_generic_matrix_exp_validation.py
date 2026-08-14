"""Substantive non-MLSL backend validation for the generic primitive protocol."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
from primitive_protocol import (  # noqa: E402
    PrimitiveDomain,
    audit_batch_and_device,
    audit_value_and_gradient,
    make_audit,
)


class MatrixExponentialAction:
    """Batched differentiable action of exp(t A) on a two-dimensional vector."""

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        times = inputs[..., 0]
        vectors = inputs[..., 1:]
        if parameters.ndim == 1:
            matrix = parameters.reshape(2, 2)
            exponent = torch.matrix_exp(times[..., None, None] * matrix)
        else:
            matrix = parameters.reshape(-1, 2, 2)
            exponent = torch.matrix_exp(times[:, None, None] * matrix)
        return torch.bmm(exponent, vectors[..., None]).squeeze(-1)


def scipy_reference(inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from scipy.linalg import expm

    values = []
    matrix = parameters.detach().cpu().numpy().reshape(2, 2)
    for row in inputs.detach().cpu().numpy():
        values.append(expm(float(row[0]) * matrix) @ row[1:])
    return torch.tensor(np.asarray(values), dtype=inputs.dtype, device=inputs.device)


def calibrate(backend, device, dtype):
    true_params = torch.tensor([-0.70, 0.20, -0.10, -0.40], dtype=dtype, device=device)
    times = torch.linspace(0.05, 1.5, 24, dtype=dtype, device=device)
    vectors = torch.tensor([1.0, -0.4], dtype=dtype, device=device).expand(len(times), -1)
    inputs = torch.cat((times[:, None], vectors), dim=1)
    target = backend(inputs, true_params) + 0.001 * torch.randn_like(vectors)
    estimate = torch.tensor([-0.50, 0.05, -0.05, -0.25], dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([estimate], lr=0.03)
    start = time.perf_counter()
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((backend(inputs, estimate) - target) ** 2)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "parameter_l1_error": float(torch.mean((estimate - true_params).abs()).detach().cpu()),
        "final_loss": float(loss.detach().cpu()),
        "gradient_finite": bool(torch.isfinite(estimate.grad).all().item()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def module_reuse(backend, inputs, target, parameters):
    """Use the matrix primitive as the differentiable backbone of a module."""

    class ComposableModule(torch.nn.Module):
        def __init__(self, initial):
            super().__init__()
            self.matrix = torch.nn.Parameter(initial.clone())
            self.residual = torch.nn.Linear(2, 2, bias=False, dtype=initial.dtype).to(initial.device)
            torch.nn.init.zeros_(self.residual.weight)

        def forward(self, values):
            base = backend(values, self.matrix)
            return base + self.residual(base)

    module = ComposableModule(parameters * 0.9)
    prediction = module(inputs)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    return {
        "output_shape_preserved": prediction.shape == target.shape,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "all_gradients_finite": all(grad is not None and torch.isfinite(grad).all().item() for grad in gradients),
    }


def main():
    torch.manual_seed(55201)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    backend = MatrixExponentialAction()
    parameters = torch.tensor([-0.70, 0.20, -0.10, -0.40], dtype=dtype, device=device)
    times = torch.linspace(0.03, 1.8, 32, dtype=dtype, device=device)
    vectors = torch.randn(32, 2, dtype=dtype, device=device)
    inputs = torch.cat((times[:, None], vectors), dim=1)
    reference = scipy_reference(inputs, parameters)
    value_gradient = audit_value_and_gradient(
        backend,
        inputs,
        parameters,
        reference,
        direction=torch.tensor([0.3, -0.2, 0.4, -0.5], dtype=dtype, device=device),
    )
    batch_inputs = inputs[:16]
    batch_parameters = parameters.expand(16, -1)
    batch_device = audit_batch_and_device(backend, batch_inputs, batch_parameters)
    audit = make_audit(
        "matrix_exponential_action",
        PrimitiveDomain(
            input_description="batched time-vector pairs for a 2x2 stable matrix",
            parameter_ranges={"matrix_entries": (-1.0, 0.5)},
            output_description="batched exp(t A) v vectors",
            supports_batch=True,
            supports_gpu=torch.cuda.is_available(),
            supports_autograd=True,
        ),
        value_gradient,
        batch_device,
        warnings=["2x2 stable-matrix domain; not a general matrix-function benchmark"],
    )
    calibration = calibrate(backend, device, dtype)
    reuse = module_reuse(backend, inputs, reference, parameters)
    result = {
        "backend_audit": audit.to_dict(),
        "calibration": calibration,
        "module_reuse": reuse,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "interpretation": "second backend validation for cross-primitive protocol; not an MLSL result",
    }
    out = ROOT / "P4" / "results" / "p4_generic_matrix_exp_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

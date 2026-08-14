"""Third substantive backend: differentiable RK4 ODE-step primitive."""

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


class RK4LinearStep:
    """One differentiable RK4 step for y' = A y with explicit h and y."""

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        h = inputs[..., 0]
        state = inputs[..., 1:]
        matrix = parameters.reshape(2, 2)

        def rhs(value):
            return torch.matmul(value, matrix.transpose(-1, -2))

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * h[..., None] * k1)
        k3 = rhs(state + 0.5 * h[..., None] * k2)
        k4 = rhs(state + h[..., None] * k3)
        return state + (h[..., None] / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def exact_reference(inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    matrix = parameters.reshape(2, 2)
    states = inputs[..., 1:]
    outputs = []
    for index, row in enumerate(inputs.detach().cpu().numpy()):
        h = float(row[0])
        exponent = torch.matrix_exp(h * matrix)
        outputs.append(exponent @ states[index])
    return torch.stack(outputs)


def calibrate(backend, inputs, target, true_params, seed):
    torch.manual_seed(seed)
    estimate = (true_params * 0.80).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=0.04)
    start = time.perf_counter()
    for _ in range(350):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((backend(inputs, estimate) - target) ** 2)
        loss.backward()
        optimizer.step()
    if inputs.is_cuda:
        torch.cuda.synchronize()
    return {
        "parameter_l1_error": float(torch.mean((estimate - true_params).abs()).detach().cpu()),
        "final_loss": float(loss.detach().cpu()),
        "gradient_finite": bool(torch.isfinite(estimate.grad).all().item()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def long_horizon(backend, matrix, device, dtype):
    matrix = matrix.reshape(2, 2)
    state = torch.tensor([1.0, -0.3], dtype=dtype, device=device)
    step = 0.05
    steps = 100
    current = state
    for _ in range(steps):
        current = backend(torch.tensor([[step, current[0], current[1]]], dtype=dtype, device=device), matrix).squeeze(0)
    exact = torch.matrix_exp((step * steps) * matrix) @ state
    return {
        "final_time": step * steps,
        "absolute_error": float(torch.max((current - exact).abs()).detach().cpu()),
        "state_finite": bool(torch.isfinite(current).all().item()),
    }


def main():
    torch.manual_seed(55401)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    backend = RK4LinearStep()
    parameters = torch.tensor([-0.70, 0.20, -0.10, -0.40], dtype=dtype, device=device)
    h = torch.linspace(0.001, 0.40, 32, dtype=dtype, device=device)
    states = torch.randn(32, 2, dtype=dtype, device=device)
    inputs = torch.cat((h[:, None], states), dim=1)
    reference = exact_reference(inputs, parameters)
    value_gradient = audit_value_and_gradient(
        backend,
        inputs,
        parameters,
        reference,
        direction=torch.tensor([0.2, -0.3, 0.4, -0.1], dtype=dtype, device=device),
    )
    batch_device = audit_batch_and_device(backend, inputs[:16], parameters)
    audit = make_audit(
        "rk4_linear_ode_step",
        PrimitiveDomain(
            input_description="batched step-size and two-dimensional states",
            parameter_ranges={"matrix_entries": (-1.0, 0.5), "step_size": (0.001, 0.40)},
            output_description="batched one-step states for y'=Ay",
            supports_batch=True,
            supports_gpu=torch.cuda.is_available(),
            supports_autograd=True,
        ),
        value_gradient,
        batch_device,
        warnings=["linear 2D ODE and fixed-step RK4 domain; not a general ODE solver benchmark"],
    )
    calibration_inputs = torch.cat(
        (
            torch.rand(48, 1, dtype=dtype, device=device) * 0.35 + 0.01,
            torch.randn(48, 2, dtype=dtype, device=device),
        ),
        dim=1,
    )
    calibration_target = backend(calibration_inputs, parameters) + 0.001 * torch.randn(48, 2, dtype=dtype, device=device)
    calibration = calibrate(backend, calibration_inputs, calibration_target, parameters, 55402)
    reuse_module = torch.nn.Sequential(
        torch.nn.Linear(2, 8, dtype=dtype),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2, dtype=dtype),
    ).to(device)
    composed = backend(inputs, parameters) + reuse_module(backend(inputs, parameters))
    reuse_loss = torch.mean((composed - reference) ** 2)
    reuse_loss.backward()
    reuse = {
        "output_shape_preserved": composed.shape == reference.shape,
        "loss_finite": bool(torch.isfinite(reuse_loss).item()),
        "all_gradients_finite": all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in reuse_module.parameters()),
    }
    result = {
        "backend_audit": audit.to_dict(),
        "calibration": calibration,
        "long_horizon": long_horizon(backend, parameters, device, dtype),
        "module_reuse": reuse,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "interpretation": "third backend validation for cross-primitive protocol; not an MLSL result",
    }
    out = ROOT / "P4" / "results" / "p4_generic_ode_step_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

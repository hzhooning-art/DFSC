"""Second ODE family: differentiable RK4 step for the logistic equation."""

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


class LogisticRK4Step:
    """One differentiable RK4 step for y' = r*y*(1-y/K)."""

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        h = inputs[..., 0]
        state = inputs[..., 1]
        rate, capacity = parameters.unbind(dim=-1) if parameters.ndim > 1 else parameters.unbind()

        def rhs(value):
            return rate * value * (1.0 - value / capacity)

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * h * k1)
        k3 = rhs(state + 0.5 * h * k2)
        k4 = rhs(state + h * k3)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def exact_reference(inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    rate, capacity = parameters.unbind()
    h = inputs[..., 0]
    state = inputs[..., 1]
    exp_term = torch.exp(rate * h)
    return capacity * state * exp_term / (capacity + state * (exp_term - 1.0))


def calibrate(backend, inputs, target, true_params, seed):
    torch.manual_seed(seed)
    raw = torch.tensor([float(true_params[0] * 0.75), float(true_params[1] * 0.85)], dtype=true_params.dtype, device=true_params.device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.025)
    start = time.perf_counter()
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        # Positivity is enforced for the physical parameters.
        estimate = torch.stack((torch.nn.functional.softplus(raw[0]), torch.nn.functional.softplus(raw[1]) + 0.5))
        loss = torch.mean((backend(inputs, estimate) - target) ** 2)
        loss.backward()
        optimizer.step()
    if inputs.is_cuda:
        torch.cuda.synchronize()
    return {
        "parameter_l1_error": float(torch.mean((estimate - true_params).abs()).detach().cpu()),
        "final_loss": float(loss.detach().cpu()),
        "gradient_finite": bool(torch.isfinite(raw.grad).all().item()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def repeated_step(backend, parameters, initial, step, steps):
    current = initial
    for _ in range(steps):
        row = torch.stack((torch.as_tensor(step, dtype=parameters.dtype, device=parameters.device), current)).reshape(1, 2)
        current = backend(row, parameters).squeeze(0)
    return current


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    backend = LogisticRK4Step()
    rows = []
    for seed in [55600, 55601, 55602, 55603, 55604]:
        torch.manual_seed(seed)
        true_params = torch.stack(
            (
                0.70 + 0.15 * torch.rand((), dtype=dtype, device=device),
                1.5 + 1.0 * torch.rand((), dtype=dtype, device=device),
            )
        )
        h = torch.linspace(0.001, 0.25, 32, dtype=dtype, device=device)
        states = 0.1 + 0.8 * torch.rand(32, dtype=dtype, device=device)
        inputs = torch.cat((h[:, None], states[:, None]), dim=1)
        reference = exact_reference(inputs, true_params)
        prediction = backend(inputs, true_params)
        audit = audit_value_and_gradient(
            backend,
            inputs,
            true_params,
            reference,
            direction=torch.tensor([0.2, -0.3], dtype=dtype, device=device),
        )
        batch_device = audit_batch_and_device(backend, inputs[:16], true_params)
        ood_h = torch.linspace(0.26, 0.50, 24, dtype=dtype, device=device)
        ood_states = 0.1 + 0.8 * torch.rand(24, dtype=dtype, device=device)
        ood_inputs = torch.cat((ood_h[:, None], ood_states[:, None]), dim=1)
        ood_reference = exact_reference(ood_inputs, true_params)
        ood_prediction = backend(ood_inputs, true_params)
        initial = torch.tensor(0.15, dtype=dtype, device=device)
        long_errors = {}
        for step in (0.05, 0.20):
            steps = int(10.0 / step)
            numerical = repeated_step(backend, true_params, initial, step, steps)
            exact = exact_reference(torch.tensor([[10.0, initial]], dtype=dtype, device=device), true_params).squeeze(0)
            long_errors[str(step)] = {
                "absolute_error": float(torch.abs(numerical - exact).detach().cpu()),
                "state_finite": bool(torch.isfinite(numerical).item()),
            }
        calibration_inputs = torch.cat(
            (
                torch.rand(64, 1, dtype=dtype, device=device) * 0.22 + 0.01,
                0.1 + 0.8 * torch.rand(64, 1, dtype=dtype, device=device),
            ),
            dim=1,
        )
        calibration_target = backend(calibration_inputs, true_params) + 0.001 * torch.randn(64, dtype=dtype, device=device)
        calibration = calibrate(backend, calibration_inputs, calibration_target, true_params, seed + 100)
        residual_module = torch.nn.Sequential(
            torch.nn.Linear(1, 8, dtype=dtype), torch.nn.Tanh(), torch.nn.Linear(8, 1, dtype=dtype)
        ).to(device)
        composed = backend(inputs, true_params)[:, None] + residual_module(prediction[:, None])
        reuse_loss = torch.mean((composed - reference[:, None]) ** 2)
        reuse_loss.backward()
        reuse = {
            "output_shape_preserved": composed.shape == reference[:, None].shape,
            "loss_finite": bool(torch.isfinite(reuse_loss).item()),
            "all_gradients_finite": all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in residual_module.parameters()),
        }
        rows.append(
            {
                "seed": seed,
                "in_domain_max_abs_error": float(torch.max((prediction - reference).abs()).detach().cpu()),
                "ood_max_abs_error": float(torch.max((ood_prediction - ood_reference).abs()).detach().cpu()),
                "ood_values_finite": bool(torch.isfinite(ood_prediction).all().item()),
                "gradient_directional_relative_error": audit["gradient_directional_relative_error"],
                "gradient_finite": audit["gradient_finite"],
                "batch_device": batch_device,
                "long_horizon": long_errors,
                "calibration": calibration,
                "module_reuse": reuse,
            }
        )

    def mean_std(values):
        tensor = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=True))}

    result = {
        "backend": "logistic_rk4_step",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": [55600, 55601, 55602, 55603, 55604],
        "in_domain_step_range": [0.001, 0.25],
        "ood_step_range": [0.26, 0.50],
        "rows": rows,
        "summary": {
            "in_domain_max_abs_error": mean_std([row["in_domain_max_abs_error"] for row in rows]),
            "ood_max_abs_error": mean_std([row["ood_max_abs_error"] for row in rows]),
            "gradient_directional_relative_error": mean_std([row["gradient_directional_relative_error"] for row in rows]),
            "calibration_parameter_l1_error": mean_std([row["calibration"]["parameter_l1_error"] for row in rows]),
            "long_horizon_step_0.05_error": mean_std([row["long_horizon"]["0.05"]["absolute_error"] for row in rows]),
            "long_horizon_step_0.20_error": mean_std([row["long_horizon"]["0.2"]["absolute_error"] for row in rows]),
        },
        "all_ood_values_finite": all(row["ood_values_finite"] for row in rows),
        "all_gradients_finite": all(row["gradient_finite"] and row["calibration"]["gradient_finite"] for row in rows),
        "all_module_reuse_gradients_finite": all(row["module_reuse"]["all_gradients_finite"] for row in rows),
        "interpretation": "second ODE family cross-primitive evidence; logistic fixed-step RK4 domain only",
    }
    out = ROOT / "P4" / "results" / "p4_nonlinear_ode_step_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

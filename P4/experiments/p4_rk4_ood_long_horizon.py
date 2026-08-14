"""Multi-seed OOD and long-horizon audit for the differentiable RK4 backend."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
from primitive_protocol import audit_value_and_gradient  # noqa: E402
from p4_generic_ode_step_validation import RK4LinearStep  # noqa: E402


def make_stable_matrix(seed, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    diagonal = -0.20 - 0.80 * torch.rand(2, generator=generator, device=device, dtype=dtype)
    coupling = -0.20 + 0.40 * torch.rand(2, generator=generator, device=device, dtype=dtype)
    return torch.stack((diagonal[0], coupling[0], coupling[1], diagonal[1]))


def exact_step(inputs, parameters):
    matrix = parameters.reshape(2, 2)
    states = inputs[..., 1:]
    return torch.stack([torch.matrix_exp(float(row[0]) * matrix) @ states[index] for index, row in enumerate(inputs.detach().cpu().numpy())])


def repeated_step(backend, matrix, initial, step, steps):
    current = initial
    for _ in range(steps):
        row = torch.tensor([[step, current[0], current[1]]], dtype=matrix.dtype, device=matrix.device)
        current = backend(row, matrix).squeeze(0)
    return current


def calibrate(backend, inputs, target, true_params, seed):
    torch.manual_seed(seed)
    estimate = (true_params * 0.75).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=0.035)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    backend = RK4LinearStep()
    rows = []
    for seed in [55500, 55501, 55502, 55503, 55504]:
        torch.manual_seed(seed)
        matrix = make_stable_matrix(seed, device, dtype)
        # In-domain single-step checks.
        h = torch.linspace(0.001, 0.40, 32, dtype=dtype, device=device)
        states = torch.randn(32, 2, dtype=dtype, device=device)
        inputs = torch.cat((h[:, None], states), dim=1)
        reference = exact_step(inputs, matrix)
        prediction = backend(inputs, matrix)
        audit = audit_value_and_gradient(
            backend,
            inputs,
            matrix,
            reference,
            direction=torch.tensor([0.2, -0.3, 0.4, -0.1], dtype=dtype, device=device),
        )
        # OOD step-size check beyond the calibration range.
        ood_h = torch.linspace(0.41, 0.80, 24, dtype=dtype, device=device)
        ood_states = torch.randn(24, 2, dtype=dtype, device=device)
        ood_inputs = torch.cat((ood_h[:, None], ood_states), dim=1)
        ood_reference = exact_step(ood_inputs, matrix)
        ood_prediction = backend(ood_inputs, matrix)
        # Long-horizon comparisons at two step sizes.
        initial = torch.tensor([1.0, -0.3], dtype=dtype, device=device)
        long_errors = {}
        for step in (0.05, 0.20):
            steps = int(5.0 / step)
            numerical = repeated_step(backend, matrix, initial, step, steps)
            exact = torch.matrix_exp(5.0 * matrix.reshape(2, 2)) @ initial
            long_errors[str(step)] = {
                "absolute_error": float(torch.max((numerical - exact).abs()).detach().cpu()),
                "state_finite": bool(torch.isfinite(numerical).all().item()),
            }
        # Calibration is performed on the declared in-domain step range.
        calibration_inputs = torch.cat(
            (
                torch.rand(48, 1, dtype=dtype, device=device) * 0.35 + 0.01,
                torch.randn(48, 2, dtype=dtype, device=device),
            ),
            dim=1,
        )
        calibration_target = backend(calibration_inputs, matrix) + 0.001 * torch.randn(48, 2, dtype=dtype, device=device)
        calibration = calibrate(backend, calibration_inputs, calibration_target, matrix, seed + 100)
        rows.append(
            {
                "seed": seed,
                "in_domain_max_abs_error": float(torch.max((prediction - reference).abs()).detach().cpu()),
                "ood_max_abs_error": float(torch.max((ood_prediction - ood_reference).abs()).detach().cpu()),
                "ood_values_finite": bool(torch.isfinite(ood_prediction).all().item()),
                "gradient_directional_relative_error": audit["gradient_directional_relative_error"],
                "gradient_finite": audit["gradient_finite"],
                "long_horizon": long_errors,
                "calibration": calibration,
            }
        )

    def mean_std(values):
        tensor = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=True))}

    result = {
        "backend": "rk4_linear_ode_step",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": [55500, 55501, 55502, 55503, 55504],
        "in_domain_step_range": [0.001, 0.40],
        "ood_step_range": [0.41, 0.80],
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
        "interpretation": "multi-seed OOD and long-horizon evidence; fixed-step RK4 domain only",
    }
    out = ROOT / "P4" / "results" / "p4_rk4_ood_long_horizon.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

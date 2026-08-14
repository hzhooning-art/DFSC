"""Multi-seed OOD and long-horizon audit for the matrix-exponential primitive."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4"))
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from primitive_protocol import audit_value_and_gradient  # noqa: E402
from p4_generic_matrix_exp_validation import MatrixExponentialAction, scipy_reference  # noqa: E402


def make_stable_matrix(seed, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    diagonal = -0.25 - 0.75 * torch.rand(2, generator=generator, device=device, dtype=dtype)
    coupling = -0.25 + 0.50 * torch.rand(2, generator=generator, device=device, dtype=dtype)
    return torch.stack((diagonal[0], coupling[0], coupling[1], diagonal[1]))


def calibrate(backend, inputs, target, true_params, seed):
    torch.manual_seed(seed)
    estimate = (true_params * 0.70).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=0.025)
    start = time.perf_counter()
    for _ in range(400):
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
    backend = MatrixExponentialAction()
    times = torch.cat(
        (
            torch.linspace(0.02, 1.5, 24, dtype=dtype, device=device),
            torch.tensor([2.0, 4.0, 8.0, 12.0], dtype=dtype, device=device),
        )
    )
    rows = []
    for seed in [55300, 55301, 55302, 55303, 55304]:
        torch.manual_seed(seed)
        params = make_stable_matrix(seed, device, dtype)
        vectors = torch.randn(len(times), 2, dtype=dtype, device=device)
        inputs = torch.cat((times[:, None], vectors), dim=1)
        reference = scipy_reference(inputs, params)
        audit = audit_value_and_gradient(
            backend,
            inputs,
            params,
            reference,
            direction=torch.tensor([0.2, -0.3, 0.4, -0.1], dtype=dtype, device=device),
        )
        prediction = backend(inputs, params)
        abs_error = (prediction - reference).abs()
        relative_error = abs_error / (reference.abs() + 1e-10)
        calibration_times = torch.linspace(0.05, 4.0, 32, dtype=dtype, device=device)
        calibration_vectors = torch.randn(32, 2, dtype=dtype, device=device)
        calibration_inputs = torch.cat((calibration_times[:, None], calibration_vectors), dim=1)
        calibration_target = backend(calibration_inputs, params) + 0.001 * torch.randn(32, 2, dtype=dtype, device=device)
        calibration = calibrate(backend, calibration_inputs, calibration_target, params, seed + 100)
        horizon_mask = times >= 4.0
        rows.append(
            {
                "seed": seed,
                "max_abs_error": float(abs_error.max().detach().cpu()),
                "max_relative_error": float(relative_error.max().detach().cpu()),
                "long_horizon_max_abs_error": float(abs_error[horizon_mask].max().detach().cpu()),
                "long_horizon_rms_error": float(torch.sqrt(torch.mean(abs_error[horizon_mask] ** 2)).detach().cpu()),
                "long_horizon_reference_rms": float(torch.sqrt(torch.mean(reference[horizon_mask] ** 2)).detach().cpu()),
                "values_finite": bool(torch.isfinite(prediction).all().item()),
                "gradient_finite": audit["gradient_finite"],
                "gradient_directional_relative_error": audit["gradient_directional_relative_error"],
                "calibration": calibration,
            }
        )
    def mean_std(values):
        tensor = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=True))}

    result = {
        "backend": "matrix_exponential_action",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": [55300, 55301, 55302, 55303, 55304],
        "time_domain": [0.02, 12.0],
        "rows": rows,
        "summary": {
            "max_abs_error": mean_std([row["max_abs_error"] for row in rows]),
            "long_horizon_max_abs_error": mean_std([row["long_horizon_max_abs_error"] for row in rows]),
            "long_horizon_rms_error": mean_std([row["long_horizon_rms_error"] for row in rows]),
            "gradient_directional_relative_error": mean_std([row["gradient_directional_relative_error"] for row in rows]),
            "calibration_parameter_l1_error": mean_std([row["calibration"]["parameter_l1_error"] for row in rows]),
        },
        "all_values_finite": all(row["values_finite"] for row in rows),
        "all_gradients_finite": all(row["gradient_finite"] and row["calibration"]["gradient_finite"] for row in rows),
        "interpretation": "multi-seed OOD and long-horizon evidence; not a universal matrix-function claim",
    }
    out = ROOT / "P4" / "results" / "p4_matrix_exp_ood_long_horizon.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

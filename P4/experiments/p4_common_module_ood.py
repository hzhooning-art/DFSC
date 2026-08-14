"""Common module-level OOD task for two distinct primitive backends."""

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
from p4_nonlinear_ode_step_validation import LogisticRK4Step  # noqa: E402


def matrix_params_from_raw(raw):
    values = torch.tanh(raw)
    return torch.stack((-0.10 - 0.90 * torch.sigmoid(raw[..., 0]), 0.30 * values[..., 1], 0.30 * values[..., 2], -0.10 - 0.90 * torch.sigmoid(raw[..., 3])), dim=-1)


def logistic_params_from_raw(raw):
    return torch.stack((0.40 + 1.00 * torch.sigmoid(raw[..., 0]), 1.00 + 2.00 * torch.sigmoid(raw[..., 1])), dim=-1)


def matrix_propagate(backend, y0, target_time, params):
    inputs = torch.cat((target_time[:, None], y0), dim=1)
    return backend(inputs, params)


def logistic_propagate(backend, y0, target_time, params):
    current = y0 if y0.ndim == 2 else y0[:, None]
    steps = 20
    h = target_time / steps
    for _ in range(steps):
        current = backend(torch.cat((h[:, None], current), dim=1), params)[:, None]
    return current


def make_dataset(kind, n, device, dtype, seed, ood=False, noise_sigma=0.003, target_time_value=1.0):
    generator = torch.Generator(device=device).manual_seed(seed)
    if kind == "matrix":
        params = torch.stack(
            (
                -0.10 - (0.90 if ood else 0.75) * torch.rand(n, generator=generator, device=device, dtype=dtype),
                (-0.30 if ood else -0.20) + (0.60 if ood else 0.40) * torch.rand(n, generator=generator, device=device, dtype=dtype),
                (-0.30 if ood else -0.20) + (0.60 if ood else 0.40) * torch.rand(n, generator=generator, device=device, dtype=dtype),
                -0.10 - (0.90 if ood else 0.75) * torch.rand(n, generator=generator, device=device, dtype=dtype),
            ),
            dim=1,
        )
        y0 = torch.randn(n, 2, generator=generator, device=device, dtype=dtype)
        times = torch.tensor([0.05, 0.10, 0.15], dtype=dtype, device=device)
        context = []
        backend = MatrixExponentialAction()
        for t in times:
            context.append(matrix_propagate(backend, y0, torch.full((n,), t, dtype=dtype, device=device), params))
        context = torch.cat((y0, *context), dim=1)
        target_time = torch.full((n,), target_time_value, dtype=dtype, device=device)
        target = matrix_propagate(backend, y0, target_time, params)
    else:
        params = torch.stack(
            (
                0.40 + (1.00 if ood else 0.70) * torch.rand(n, generator=generator, device=device, dtype=dtype),
                1.00 + (2.00 if ood else 1.50) * torch.rand(n, generator=generator, device=device, dtype=dtype),
            ),
            dim=1,
        )
        y0 = 0.05 + 0.90 * torch.rand(n, 1, generator=generator, device=device, dtype=dtype)
        backend = LogisticRK4Step()
        context = [y0]
        for t in (0.05, 0.10, 0.15):
            context.append(logistic_propagate(backend, y0, torch.full((n,), t, dtype=dtype, device=device), params))
        context = torch.cat(context, dim=1)
        target_time = torch.full((n,), target_time_value, dtype=dtype, device=device)
        target = logistic_propagate(backend, y0, target_time, params)
    context = context + noise_sigma * torch.randn_like(context)
    return context, y0, target_time, target, params


class PrimitiveForecast(torch.nn.Module):
    def __init__(self, kind, context_dim, output_dim):
        super().__init__()
        self.kind = kind
        self.encoder = torch.nn.Sequential(torch.nn.Linear(context_dim, 48), torch.nn.Tanh(), torch.nn.Linear(48, 32), torch.nn.Tanh())
        self.parameter_head = torch.nn.Linear(32, 4 if kind == "matrix" else 2)
        self.backend = MatrixExponentialAction() if kind == "matrix" else LogisticRK4Step()

    def forward(self, context, y0, target_time):
        raw = self.parameter_head(self.encoder(context))
        params = matrix_params_from_raw(raw) if self.kind == "matrix" else logistic_params_from_raw(raw)
        prediction = matrix_propagate(self.backend, y0, target_time, params) if self.kind == "matrix" else logistic_propagate(self.backend, y0, target_time, params)
        return prediction, params


class PureMLP(torch.nn.Module):
    def __init__(self, context_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(context_dim, 48), torch.nn.Tanh(), torch.nn.Linear(48, 32), torch.nn.Tanh(), torch.nn.Linear(32, output_dim))

    def forward(self, context, y0, target_time):
        return self.net(context)


def train(model, context, y0, target_time, target, steps=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context, y0, target_time)
        prediction = prediction[0] if isinstance(prediction, tuple) else prediction
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
    if context.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - start


def run_kind(kind, device, dtype, noise_sigma, target_time_value):
    rows = []
    context_dim, output_dim = (8, 2) if kind == "matrix" else (4, 1)
    for seed in [55700, 55701, 55702]:
        train_context, train_y0, train_time, train_target, _ = make_dataset(kind, 512, device, dtype, seed, False, noise_sigma, target_time_value)
        test_context, test_y0, test_time, test_target, test_params = make_dataset(kind, 256, device, dtype, seed + 100, True, noise_sigma, target_time_value)
        torch.manual_seed(seed + 3)
        primitive_model = PrimitiveForecast(kind, context_dim, output_dim).to(device).double()
        primitive_elapsed = train(primitive_model, train_context, train_y0, train_time, train_target)
        primitive_prediction, predicted_params = primitive_model(test_context, test_y0, test_time)
        torch.manual_seed(seed + 7)
        baseline = PureMLP(context_dim, output_dim).to(device).double()
        baseline_elapsed = train(baseline, train_context, train_y0, train_time, train_target)
        baseline_prediction = baseline(test_context, test_y0, test_time)
        rows.append(
            {
                "seed": seed,
                "primitive_ood_rmse": float(torch.sqrt(torch.mean((primitive_prediction - test_target) ** 2)).detach().cpu()),
                "baseline_ood_rmse": float(torch.sqrt(torch.mean((baseline_prediction - test_target) ** 2)).detach().cpu()),
                "primitive_parameter_l1_error": float(torch.mean((predicted_params - test_params).abs()).detach().cpu()),
                "primitive_training_seconds": primitive_elapsed,
                "baseline_training_seconds": baseline_elapsed,
                "primitive_gradients_finite": all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in primitive_model.parameters()),
                "baseline_gradients_finite": all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in baseline.parameters()),
            }
        )
    def mean_std(values):
        tensor = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=True))}
    return {
        "backend": kind,
        "noise_sigma": noise_sigma,
        "target_time": target_time_value,
        "summary": {
            "primitive_ood_rmse": mean_std([row["primitive_ood_rmse"] for row in rows]),
            "baseline_ood_rmse": mean_std([row["baseline_ood_rmse"] for row in rows]),
            "primitive_parameter_l1_error": mean_std([row["primitive_parameter_l1_error"] for row in rows]),
            "primitive_training_seconds": mean_std([row["primitive_training_seconds"] for row in rows]),
            "baseline_training_seconds": mean_std([row["baseline_training_seconds"] for row in rows]),
        },
        "all_gradients_finite": all(row["primitive_gradients_finite"] and row["baseline_gradients_finite"] for row in rows),
        "rows": rows,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    regimes = {}
    for target_time_value in (1.0, 1.5):
        for noise_sigma in (0.003, 0.01):
            key = f"t{target_time_value}_noise{noise_sigma}"
            regimes[key] = {
                "matrix": run_kind("matrix", device, dtype, noise_sigma, target_time_value),
                "logistic": run_kind("logistic", device, dtype, noise_sigma, target_time_value),
        }
    result = {
        "schema": "DFSC-Common-Module-OOD-v1",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_tasks": 512,
        "test_tasks": 256,
        "seeds": [55700, 55701, 55702],
        "target_times": [1.0, 1.5],
        "noise_regimes": [0.003, 0.01],
        "results": regimes,
        "interpretation": "same module-level OOD template across two observation-noise regimes; backend metrics are reported separately",
    }
    out = ROOT / "P4" / "results" / "p4_common_module_ood.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    for regime, backends in result["results"].items():
        for name, value in backends.items():
            print(regime, name, json.dumps({key: item for key, item in value.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

"""Direct conditioning diagnostics for the matrix-primitive OOD experiment.

This script repeats only the structured matrix branch of the common OOD task
and measures whether task error co-varies with latent-generator diagnostics.
The reported correlations are descriptive and are not interpreted as causal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from p4_common_module_ood import PrimitiveForecast, make_dataset, train  # noqa: E402


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if denominator <= 0:
        return float("nan")
    return float((x * y).sum().div(denominator).detach().cpu())


def ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    result = torch.empty_like(x, dtype=torch.float64)
    result[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
    return result


def diagnostics(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvectors = torch.linalg.eig(matrices).eigenvectors
    condition = torch.linalg.cond(eigenvectors).real
    adjoint = matrices.transpose(-1, -2)
    commutator = adjoint @ matrices - matrices @ adjoint
    nonnormality = torch.linalg.matrix_norm(commutator, ord="fro") / torch.clamp(
        torch.linalg.matrix_norm(matrices, ord="fro").square(), min=1.0e-15
    )
    return condition, nonnormality


def observation_diagnostics(
    params: torch.Tensor, y0: torch.Tensor, times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure local parameter observability and generator nonnormality."""

    def observations(single_params: torch.Tensor, single_y0: torch.Tensor) -> torch.Tensor:
        matrix = single_params.reshape(2, 2)
        states = [torch.matrix_exp(time * matrix) @ single_y0 for time in times]
        return torch.cat(states)

    jacobian = torch.func.vmap(torch.func.jacrev(observations), in_dims=(0, 0))(params, y0)
    singular_values = torch.linalg.svdvals(jacobian)
    sigma_min = singular_values[..., -1]
    jacobian_condition = singular_values[..., 0] / torch.clamp(sigma_min, min=1.0e-15)

    matrices = params.reshape(-1, 2, 2)
    adjoint = matrices.transpose(-1, -2)
    commutator = adjoint @ matrices - matrices @ adjoint
    nonnormality = torch.linalg.matrix_norm(commutator, ord="fro") / torch.clamp(
        torch.linalg.matrix_norm(matrices, ord="fro").square(), min=1.0e-15
    )
    return sigma_min, jacobian_condition, nonnormality


def main() -> None:
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for target_time in (1.0, 1.5):
        for noise_sigma in (0.003, 0.01):
            for seed in (55700, 55701, 55702):
                train_context, train_y0, train_time, train_target, _ = make_dataset(
                    "matrix", 512, device, torch.float64, seed, False, noise_sigma, target_time
                )
                test_context, test_y0, test_time, test_target, test_params = make_dataset(
                    "matrix", 256, device, torch.float64, seed + 100, True, noise_sigma, target_time
                )
                torch.manual_seed(seed + 3)
                model = PrimitiveForecast("matrix", 8, 2).to(device).double()
                train(model, train_context, train_y0, train_time, train_target)
                with torch.no_grad():
                    prediction, predicted_params = model(test_context, test_y0, test_time)
                    sample_error = torch.linalg.vector_norm(prediction - test_target, dim=1)
                    parameter_error = torch.mean((predicted_params - test_params).abs(), dim=1)
                    matrices = predicted_params.reshape(-1, 2, 2)
                    condition, nonnormality = diagnostics(matrices)
                    observation_times = torch.tensor([0.05, 0.10, 0.15], dtype=torch.float64, device=device)
                    sigma_min, observation_condition, _ = observation_diagnostics(
                        predicted_params, test_y0, observation_times
                    )
                    finite = (
                        torch.isfinite(condition)
                        & torch.isfinite(nonnormality)
                        & torch.isfinite(sigma_min)
                        & torch.isfinite(observation_condition)
                        & torch.isfinite(sample_error)
                    )
                    log_condition = torch.log10(torch.clamp(condition[finite], min=1.0))
                    log_sigma_min = torch.log10(torch.clamp(sigma_min[finite], min=1.0e-15))
                    log_observation_condition = torch.log10(
                        torch.clamp(observation_condition[finite], min=1.0)
                    )
                    log_error = torch.log10(torch.clamp(sample_error[finite], min=1.0e-15))
                    log_parameter_error = torch.log10(torch.clamp(parameter_error[finite], min=1.0e-15))
                    nonnormal = nonnormality[finite]
                    rows.append(
                        {
                            "seed": seed,
                            "target_time": target_time,
                            "noise_sigma": noise_sigma,
                            "samples": int(finite.sum().item()),
                            "ood_rmse": float(torch.sqrt(torch.mean((prediction - test_target).square())).cpu()),
                            "median_eigenvector_condition": float(condition[finite].median().cpu()),
                            "p95_eigenvector_condition": float(torch.quantile(condition[finite], 0.95).cpu()),
                            "median_observation_sigma_min": float(sigma_min[finite].median().cpu()),
                            "median_observation_jacobian_condition": float(
                                observation_condition[finite].median().cpu()
                            ),
                            "median_nonnormality": float(nonnormal.median().cpu()),
                            "pearson_log_condition_vs_state_error": pearson(log_condition, log_error),
                            "spearman_condition_vs_state_error": pearson(ranks(log_condition), ranks(log_error)),
                            "pearson_log_observation_sigma_min_vs_state_error": pearson(
                                log_sigma_min, log_error
                            ),
                            "spearman_observation_sigma_min_vs_state_error": pearson(
                                ranks(log_sigma_min), ranks(log_error)
                            ),
                            "pearson_log_observation_condition_vs_state_error": pearson(
                                log_observation_condition, log_error
                            ),
                            "pearson_nonnormality_vs_state_error": pearson(nonnormal, log_error),
                            "pearson_parameter_error_vs_state_error": pearson(log_parameter_error, log_error),
                        }
                    )

    result = {
        "schema": "DFSC-P4-Matrix-OOD-Conditioning-v2",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seeds": [55700, 55701, 55702],
        "train_tasks": 512,
        "test_tasks": 256,
        "training_steps": 300,
        "rows": rows,
        "interpretation": (
            "Correlations diagnose associations among local observation sensitivity, inferred-generator "
            "conditioning, nonnormality, and OOD error. They do not establish a causal mechanism or a "
            "universal acceptance threshold."
        ),
    }
    out = ROOT / "P4" / "results" / "p4_matrix_ood_conditioning.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

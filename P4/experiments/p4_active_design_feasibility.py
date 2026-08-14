"""First feasibility test for error-aware active observation design."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from eafi_feasibility import grid_predictions, ml_family  # noqa: E402


def sensitivities(times, alpha, rate, terms):
    rows = []
    values = []
    for t in times:
        a = torch.tensor(alpha, dtype=torch.float64, device=times.device, requires_grad=True)
        r = torch.tensor(rate, dtype=torch.float64, device=times.device, requires_grad=True)
        value = ml_family(a, r, t.reshape(1), terms=terms).squeeze()
        ga, gr = torch.autograd.grad(value, (a, r), retain_graph=False)
        rows.append(torch.stack((ga, gr)))
        values.append(value)
    return torch.stack(values), torch.stack(rows)


def greedy_design(jacobian, weights, budget):
    info = torch.zeros((2, 2), dtype=torch.float64, device=jacobian.device)
    selected = []
    remaining = set(range(jacobian.shape[0]))
    for _ in range(budget):
        best_idx, best_gain = None, None
        for idx in remaining:
            candidate = info + weights[idx] * torch.outer(jacobian[idx], jacobian[idx])
            gain = torch.linalg.slogdet(candidate + 1e-8 * torch.eye(2, dtype=torch.float64, device=jacobian.device))[1]
            if best_gain is None or gain > best_gain:
                best_idx, best_gain = idx, gain
        selected.append(best_idx)
        info = info + weights[best_idx] * torch.outer(jacobian[best_idx], jacobian[best_idx])
        remaining.remove(best_idx)
    return selected


def evaluate_strategy(name, indices, grid_prediction, alpha_grid, rate_grid, reference, sigma, seeds):
    selected = grid_prediction[:, :, indices]
    ia = torch.argmin(torch.abs(alpha_grid - 0.773))
    ir = torch.argmin(torch.abs(rate_grid - 0.637))
    errors = []
    covered = []
    areas = []
    for seed in seeds:
        torch.manual_seed(seed)
        y = reference[indices] + sigma * torch.randn(len(indices), dtype=torch.float64, device=reference.device)
        loss = torch.sum((selected - y[None, None, :]) ** 2, dim=-1)
        best = torch.argmin(loss)
        bi, bj = torch.unravel_index(best, loss.shape)
        errors.append([abs(float(alpha_grid[bi] - 0.773)), abs(float(rate_grid[bj] - 0.637))])
        threshold = 2.0 * len(indices) * sigma**2
        keep = loss <= loss.min() + threshold
        covered.append(bool(keep[ia, ir].item()))
        areas.append(float(keep.float().mean().item()))
    return {
        "strategy": name,
        "indices": indices,
        "alpha_abs_error_mean": sum(e[0] for e in errors) / len(errors),
        "rate_abs_error_mean": sum(e[1] for e in errors) / len(errors),
        "joint_error_mean": sum((e[0] ** 2 + e[1] ** 2) ** 0.5 for e in errors) / len(errors),
        "profile_coverage": sum(covered) / len(covered),
        "profile_area_fraction_mean": sum(areas) / len(areas),
        "replicates": len(errors),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_times = torch.linspace(0.05, 2.0, 64, dtype=torch.float64, device=device)
    budget = 8
    nominal_alpha, nominal_rate, sigma = 0.78, 0.65, 0.01
    low_values, low_jac = sensitivities(candidate_times, nominal_alpha, nominal_rate, terms=8)
    high_values, high_jac = sensitivities(candidate_times, nominal_alpha, nominal_rate, terms=100)
    value_error = (high_values - low_values).abs()
    gradient_error = (high_jac - low_jac).norm(dim=1)
    # Standard design trusts the high-accuracy sensitivity; EAD penalizes local
    # value/gradient uncertainty while retaining the low-order runtime model.
    standard_weights = torch.full_like(value_error, 1.0 / sigma**2)
    error_aware_weights = 1.0 / (sigma**2 + value_error**2 + gradient_error**2)
    standard_indices = greedy_design(high_jac, standard_weights, budget)
    error_aware_indices = greedy_design(low_jac, error_aware_weights, budget)
    uniform_indices = torch.linspace(0, len(candidate_times) - 1, budget).round().long().tolist()

    alpha_grid = torch.linspace(0.65, 0.91, 41, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 41, dtype=torch.float64, device=device)
    fit_times = candidate_times
    reference = ml_family(torch.tensor(0.773, dtype=torch.float64, device=device), torch.tensor(0.637, dtype=torch.float64, device=device), fit_times, terms=100)
    fit_prediction = grid_predictions(alpha_grid, rate_grid, fit_times, terms=100)
    seeds = range(47000, 47040)
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "candidate_count": len(candidate_times),
        "budget": budget,
        "nominal": {"alpha": nominal_alpha, "rate": nominal_rate, "sigma": sigma},
        "selected_times": {
            "uniform": [float(candidate_times[i].cpu()) for i in uniform_indices],
            "standard_d_opt": [float(candidate_times[i].cpu()) for i in standard_indices],
            "error_aware_d_opt": [float(candidate_times[i].cpu()) for i in error_aware_indices],
        },
        "design_diagnostics": {
            "value_error_max": float(value_error.max().cpu()),
            "gradient_error_max": float(gradient_error.max().cpu()),
            "standard_logdet": float(torch.linalg.slogdet(torch.eye(2, dtype=torch.float64, device=device) * 1e-8 + sum((torch.outer(high_jac[i], high_jac[i]) for i in standard_indices)))[1].cpu()),
            "error_aware_logdet": float(torch.linalg.slogdet(torch.eye(2, dtype=torch.float64, device=device) * 1e-8 + sum((error_aware_weights[i] * torch.outer(low_jac[i], low_jac[i]) for i in error_aware_indices)))[1].cpu()),
        },
        "results": [
            evaluate_strategy("uniform", uniform_indices, fit_prediction, alpha_grid, rate_grid, reference, sigma, seeds),
            evaluate_strategy("standard_d_opt", standard_indices, fit_prediction, alpha_grid, rate_grid, reference, sigma, seeds),
            evaluate_strategy("error_aware_d_opt", error_aware_indices, fit_prediction, alpha_grid, rate_grid, reference, sigma, seeds),
        ],
        "interpretation": "controlled feasibility result; design objective and hyperparameters are not yet frozen for publication",
    }
    out = ROOT / "P4" / "results" / "p4_active_design_feasibility.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Feasibility study for error-aware fractional identification (EAFI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P1" / "paper1_mlsl"))
from dfsc.mittag_leffler import mittag_leffler_e  # noqa: E402


def ml_family(alpha, rate, times, terms=100):
    return mittag_leffler_e(
        alpha,
        -rate * times.pow(alpha),
        terms=terms,
        custom_backward=False,
        method="series",
    )


def grid_predictions(alpha_grid, rate_grid, times, terms):
    rows = []
    for a in alpha_grid:
        for r in rate_grid:
            rows.append(ml_family(a, r, times, terms=terms))
    return torch.stack(rows).reshape(len(alpha_grid), len(rate_grid), -1)


def profile_interval(loss, threshold):
    minimum = loss.min()
    keep = loss <= minimum + threshold
    return keep, float(keep.sum().item())


def run_known_truth(device):
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    true_a, true_r, sigma = 0.78, 0.65, 0.01
    alpha_grid = torch.linspace(0.65, 0.91, 27, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 26, dtype=torch.float64, device=device)
    reference = ml_family(torch.tensor(true_a, dtype=torch.float64, device=device), torch.tensor(true_r, dtype=torch.float64, device=device), times, terms=100)
    pred_high = grid_predictions(alpha_grid, rate_grid, times, terms=100)
    pred_low = grid_predictions(alpha_grid, rate_grid, times, terms=4)
    numerical_floor = torch.mean((pred_high - pred_low) ** 2, dim=-1).max()
    records = []
    for seed in range(20):
        torch.manual_seed(1000 + seed)
        y = reference + sigma * torch.randn_like(reference)
        loss = torch.sum((pred_low - y[None, None, :]) ** 2, dim=-1)
        standard_keep, standard_width = profile_interval(loss, 2.0 * times.numel() * sigma**2)
        aware_threshold = 2.0 * times.numel() * sigma**2 + 3.0 * times.numel() * numerical_floor
        aware_keep, aware_width = profile_interval(loss, aware_threshold)
        true_cell = torch.argmin(torch.abs(alpha_grid - true_a))
        true_rate_cell = torch.argmin(torch.abs(rate_grid - true_r))
        records.append({
            "standard_covered": bool(standard_keep[true_cell, true_rate_cell].item()),
            "aware_covered": bool(aware_keep[true_cell, true_rate_cell].item()),
            "standard_width": standard_width,
            "aware_width": aware_width,
        })
    return {
        "true_alpha": true_a,
        "true_rate": true_r,
        "noise_sigma": sigma,
        "numerical_floor_mse": float(numerical_floor.cpu()),
        "replicates": len(records),
        "standard_coverage": sum(r["standard_covered"] for r in records) / len(records),
        "aware_coverage": sum(r["aware_covered"] for r in records) / len(records),
        "standard_width_mean": sum(r["standard_width"] for r in records) / len(records),
        "aware_width_mean": sum(r["aware_width"] for r in records) / len(records),
    }


def run_mismatch(device):
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    sigma = 0.01
    alpha_grid = torch.linspace(0.65, 0.91, 27, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 26, dtype=torch.float64, device=device)
    pred = grid_predictions(alpha_grid, rate_grid, times, terms=100)
    # Deliberately misspecified stretched-exponential observations.
    y0 = torch.exp(-0.95 * times.pow(0.30))
    accepted = 0
    scores = []
    for seed in range(20):
        torch.manual_seed(2000 + seed)
        y = y0 + sigma * torch.randn_like(y0)
        loss = torch.sum((pred - y[None, None, :]) ** 2, dim=-1)
        best = float(loss.min().cpu())
        score = best / (times.numel() * sigma**2)
        scores.append(score)
        if score <= 3.0:
            accepted += 1
    return {
        "replicates": 20,
        "best_fit_normalized_loss_mean": sum(scores) / len(scores),
        "best_fit_normalized_loss_min": min(scores),
        "ordinary_acceptance_rate": accepted / 20,
        "eafi_abstention_rate": 1.0 - accepted / 20,
        "interpretation": "screening diagnostic; threshold is provisional and must be frozen before external validation",
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.tensor(0.78, dtype=torch.float64, device=device, requires_grad=True)
    t = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    probe = ml_family(a, torch.tensor(0.65, dtype=torch.float64, device=device), t).sum()
    grad = torch.autograd.grad(probe, a)[0]
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gradient_finite": bool(torch.isfinite(grad).item()),
        "alpha_gradient_abs": float(grad.abs().cpu()),
        "known_truth": run_known_truth(device),
        "structural_mismatch": run_mismatch(device),
        "interpretation": "P4 EAFI feasibility gate; thresholds are provisional, not publication claims",
    }
    out = ROOT / "P4" / "results" / "p4_eafi_feasibility.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

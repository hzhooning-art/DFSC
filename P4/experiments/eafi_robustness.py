"""Stress test EAFI with off-grid truth, estimated noise, and near mismatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from eafi_feasibility import grid_predictions, ml_family  # noqa: E402


def estimate_sigma(y):
    # A deliberately simple noise diagnostic: robust scale of first differences.
    d = y[1:] - y[:-1]
    centered = d - d.median()
    return (centered.abs().median() / (0.6745 * 2.0**0.5)).clamp_min(1e-4)


def run(device):
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    alpha_grid = torch.linspace(0.65, 0.91, 27, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 26, dtype=torch.float64, device=device)
    true_alpha, true_rate, true_sigma = 0.773, 0.637, 0.01
    ref = ml_family(torch.tensor(true_alpha, dtype=torch.float64, device=device), torch.tensor(true_rate, dtype=torch.float64, device=device), times, terms=100)
    pred = grid_predictions(alpha_grid, rate_grid, times, terms=8)
    floor = torch.mean((grid_predictions(alpha_grid, rate_grid, times, terms=100) - pred) ** 2, dim=-1).max()
    ia = torch.argmin(torch.abs(alpha_grid - true_alpha))
    ir = torch.argmin(torch.abs(rate_grid - true_rate))
    rows = []
    for seed in range(43000, 43040):
        torch.manual_seed(seed)
        y = ref + true_sigma * torch.randn_like(ref)
        sigma_hat = estimate_sigma(y)
        loss = torch.sum((pred - y[None, None, :]) ** 2, dim=-1)
        standard = loss <= loss.min() + 2.0 * times.numel() * sigma_hat**2
        aware = loss <= loss.min() + 2.0 * times.numel() * sigma_hat**2 + 0.25 * times.numel() * floor
        rows.append({
            "sigma_hat": float(sigma_hat.cpu()),
            "standard_covered": bool(standard[ia, ir].item()),
            "aware_covered": bool(aware[ia, ir].item()),
            "standard_area": float(standard.float().mean().item()),
            "aware_area": float(aware.float().mean().item()),
        })
    near_mismatch = []
    # Ten percent stretched-exponential contamination is intentionally subtle.
    stretched = torch.exp(-0.95 * times.pow(0.55))
    mixed = 0.90 * ref + 0.10 * stretched
    for seed in range(44000, 44040):
        torch.manual_seed(seed)
        y = mixed + true_sigma * torch.randn_like(mixed)
        sigma_hat = estimate_sigma(y)
        loss = torch.sum((pred - y[None, None, :]) ** 2, dim=-1)
        best = loss.min()
        ordinary_accept = bool((best <= 3.0 * times.numel() * sigma_hat**2).item())
        aware_accept = bool((best <= 3.0 * times.numel() * sigma_hat**2 + 0.25 * times.numel() * floor).item())
        near_mismatch.append({"sigma_hat": float(sigma_hat.cpu()), "ordinary_accept": ordinary_accept, "aware_accept": aware_accept, "normalized_best_loss": float((best / (times.numel() * sigma_hat**2)).cpu())})
    return {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "truth": {"alpha": true_alpha, "rate": true_rate, "sigma": true_sigma},
        "evaluator_terms": 8,
        "floor_mse": float(floor.cpu()),
        "off_grid": {
            "replicates": len(rows),
            "sigma_hat_mean": sum(r["sigma_hat"] for r in rows) / len(rows),
            "standard_coverage": sum(r["standard_covered"] for r in rows) / len(rows),
            "aware_coverage": sum(r["aware_covered"] for r in rows) / len(rows),
            "standard_area_mean": sum(r["standard_area"] for r in rows) / len(rows),
            "aware_area_mean": sum(r["aware_area"] for r in rows) / len(rows),
        },
        "near_mismatch": {
            "contamination_fraction": 0.10,
            "replicates": len(near_mismatch),
            "sigma_hat_mean": sum(r["sigma_hat"] for r in near_mismatch) / len(near_mismatch),
            "normalized_best_loss_mean": sum(r["normalized_best_loss"] for r in near_mismatch) / len(near_mismatch),
            "ordinary_acceptance_rate": sum(r["ordinary_accept"] for r in near_mismatch) / len(near_mismatch),
            "aware_acceptance_rate": sum(r["aware_accept"] for r in near_mismatch) / len(near_mismatch),
        },
        "interpretation": "stress-test only; sigma estimator and threshold are not frozen publication components",
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run(device)
    out = ROOT / "P4" / "results" / "p4_eafi_robustness.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

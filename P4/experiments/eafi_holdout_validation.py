"""Hold-out validation for the EAFI profile-set uncertainty rule.

The calibration seeds select an inflation factor; independent seeds report
coverage and normalized profile-set area.  This avoids reusing the same
replicates for tuning and evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "P4" / "experiments"))
from eafi_feasibility import grid_predictions, ml_family  # noqa: E402


def evaluate(device, low_terms, sigma, multiplier, seeds):
    times = torch.linspace(0.05, 1.5, 32, dtype=torch.float64, device=device)
    true_alpha, true_rate = 0.78, 0.65
    alpha_grid = torch.linspace(0.65, 0.91, 27, dtype=torch.float64, device=device)
    rate_grid = torch.linspace(0.40, 0.90, 26, dtype=torch.float64, device=device)
    reference = ml_family(
        torch.tensor(true_alpha, dtype=torch.float64, device=device),
        torch.tensor(true_rate, dtype=torch.float64, device=device),
        times,
        terms=100,
    )
    prediction = grid_predictions(alpha_grid, rate_grid, times, terms=low_terms)
    high_prediction = grid_predictions(alpha_grid, rate_grid, times, terms=100)
    floor_mse = torch.mean((high_prediction - prediction) ** 2, dim=-1).max()
    noise_budget = 2.0 * times.numel() * sigma**2
    threshold = noise_budget + multiplier * times.numel() * floor_mse
    ia = torch.argmin(torch.abs(alpha_grid - true_alpha))
    ir = torch.argmin(torch.abs(rate_grid - true_rate))
    total = len(alpha_grid) * len(rate_grid)
    records = []
    for seed in seeds:
        torch.manual_seed(seed)
        y = reference + sigma * torch.randn_like(reference)
        loss = torch.sum((prediction - y[None, None, :]) ** 2, dim=-1)
        keep = loss <= loss.min() + threshold
        kept = keep.nonzero(as_tuple=False)
        alpha_width = (alpha_grid[kept[:, 0]].max() - alpha_grid[kept[:, 0]].min()).item()
        rate_width = (rate_grid[kept[:, 1]].max() - rate_grid[kept[:, 1]].min()).item()
        records.append({
            "covered": bool(keep[ia, ir].item()),
            "area_fraction": float(keep.float().mean().item()),
            "alpha_width": alpha_width,
            "rate_width": rate_width,
        })
    return {
        "low_terms": low_terms,
        "sigma": sigma,
        "multiplier": multiplier,
        "floor_mse": float(floor_mse.cpu()),
        "threshold": float(threshold.cpu()),
        "coverage": sum(r["covered"] for r in records) / len(records),
        "area_fraction_mean": sum(r["area_fraction"] for r in records) / len(records),
        "area_fraction_std": float(torch.tensor([r["area_fraction"] for r in records], dtype=torch.float64).std(unbiased=True)),
        "alpha_width_mean": sum(r["alpha_width"] for r in records) / len(records),
        "rate_width_mean": sum(r["rate_width"] for r in records) / len(records),
        "n_seeds": len(records),
        "grid_cells": total,
    }


def select_multiplier(rows):
    # Prefer the narrowest calibration rule that reaches empirical 90% coverage.
    eligible = [r for r in rows if r["coverage"] >= 0.90]
    return min(eligible, key=lambda r: r["area_fraction_mean"]) if eligible else None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration_seeds = range(41000, 41040)
    validation_seeds = range(42000, 42040)
    multipliers = [0.25, 0.5, 1.0, 2.0, 3.0]
    configs = []
    selected = []
    for low_terms in [6, 8, 12]:
        for sigma in [0.005, 0.01, 0.02]:
            cal_rows = [evaluate(device, low_terms, sigma, m, calibration_seeds) for m in multipliers]
            choice = select_multiplier(cal_rows)
            if choice is None:
                selected.append({"low_terms": low_terms, "sigma": sigma, "status": "no_calibration_candidate"})
                continue
            val = evaluate(device, low_terms, sigma, choice["multiplier"], validation_seeds)
            selected.append({
                "low_terms": low_terms,
                "sigma": sigma,
                "selected_multiplier": choice["multiplier"],
                "calibration": choice,
                "validation": val,
                "status": "pass" if val["coverage"] >= 0.90 and val["area_fraction_mean"] <= 0.25 else "review",
            })
            configs.extend(cal_rows)
    result = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "calibration_seeds": [41000, 41039],
        "validation_seeds": [42000, 42039],
        "selection_rule": "narrowest calibration profile set with empirical coverage >= 0.90",
        "pass_rule": "validation coverage >= 0.90 and mean normalized area <= 0.25",
        "selected_configs": selected,
        "calibration_grid": configs,
        "interpretation": "hold-out evidence for an empirical EAFI uncertainty rule; not a rigorous confidence guarantee",
    }
    out = ROOT / "P4" / "results" / "p4_eafi_holdout_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"calibration_grid"}}, indent=2))


if __name__ == "__main__":
    main()
